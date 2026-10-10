package main

import (
	"fmt"
	"os"
	"path"
	"strings"
	"testing/fstest"
	"time"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/convert/safetensors"
)

// streamed is a safetensors repository opened where it lives: its JSON and
// tokenizer in memory, its shards read by range from the Hub.
type streamed struct {
	name    string // the repository's, which names the container
	meta    fstest.MapFS
	shards  []*safetensors.File
	remotes []*hf.Remote
}

// openStreamed opens r for a streamed conversion, or returns nil when r is not
// a safetensors repository -- a GGUF repository still downloads, because a
// GGUF is one file and the converter maps it.
//
// The shards are never stored: the writer asks for each tensor exactly once.
func openStreamed(cl *hf.Client, r hf.Ref) (*streamed, error) {
	pinned, files, err := cl.Pin(r)
	if err != nil {
		return nil, err
	}
	var weights []string
	for _, f := range files {
		switch {
		case strings.HasSuffix(strings.ToLower(f), ".gguf"):
			return nil, nil
		case strings.HasSuffix(f, ".safetensors") && !strings.Contains(f, "/"):
			weights = append(weights, f)
		}
	}
	if len(weights) == 0 {
		return nil, nil
	}
	// The small files are fetched into memory too: the converter reads the
	// config and tokenizer once, first, and nothing reads them again.
	s := &streamed{name: path.Base(pinned.Repo), meta: fstest.MapFS{}}
	fmt.Fprintf(os.Stderr, "stream   %s at %s: %d shard(s) and the config read in place\n",
		r, pinned.Rev[:12], len(weights))
	for _, f := range files {
		if strings.Contains(f, "/") || !small(f) {
			continue
		}
		ref := pinned
		ref.File = f
		rm, err := cl.OpenRemote(ref)
		if err != nil {
			return nil, err
		}
		// RULE 9: the size is the network's claim, so it is bounded before
		// it sizes anything. A tokenizer.json is a few MB.
		if rm.Size() > maxSmall {
			return nil, fmt.Errorf("%s is %d bytes; a config or tokenizer file over %d is "+
				"not one this converter reads", ref, rm.Size(), maxSmall)
		}
		b := make([]byte, rm.Size())
		if _, err := rm.ReadAt(b, 0); err != nil {
			return nil, err
		}
		s.meta[f] = &fstest.MapFile{Data: b}
	}
	// The index is the authority on which shards make one model, exactly as
	// for a local directory; without one, a single file is the model.
	names := weights
	if idx, ok := s.meta["model.safetensors.index.json"]; ok {
		if names, err = convert.IndexShards(idx.Data, "model.safetensors.index.json"); err != nil {
			return nil, err
		}
	} else if len(weights) != 1 {
		return nil, fmt.Errorf("%s holds %d .safetensors files and no "+
			"model.safetensors.index.json to say which belong to one model", r, len(weights))
	}
	for _, n := range names {
		ref := pinned
		ref.File = n
		rm, err := cl.OpenRemote(ref)
		if err != nil {
			return nil, err
		}
		f, err := safetensors.OpenAt(rm.Name(), rm, rm.Size())
		if err != nil {
			return nil, err
		}
		s.remotes = append(s.remotes, rm)
		s.shards = append(s.shards, f)
	}
	return s, nil
}

// maxSmall bounds one config or tokenizer file.
const maxSmall = 64 << 20

// small reports whether a repository file is one the converter reads from the
// directory: configuration, tokenizer and template. Weights, code, images and
// the model card are not fetched.
func small(name string) bool {
	switch strings.ToLower(path.Ext(name)) {
	case ".json", ".jinja", ".model", ".tiktoken", ".txt":
		return true
	}
	return false
}

// size is the shards' total length.
func (s *streamed) size() (n int64) {
	for _, r := range s.remotes {
		n += r.Size()
	}
	return n
}

// report prints how far the stream has got every interval until stop closes.
func (s *streamed) report(every time.Duration, stop <-chan struct{}) {
	total := s.size()
	t0 := time.Now()
	// The rate since the last line beside the average: the average starts
	// with the layout pass, which reads headers for minutes and few bytes.
	var last int64
	tLast := t0
	for {
		select {
		case <-stop:
			return
		case <-time.After(every):
		}
		var got int64
		for _, r := range s.remotes {
			got += r.Transferred()
		}
		el := time.Since(t0).Seconds()
		rate := float64(got) / el
		eta := "?"
		if rate > 0 {
			eta = (time.Duration(float64(total-got)/rate) * time.Second).Round(time.Minute).String()
		}
		now := time.Now()
		fmt.Fprintf(os.Stderr, "stream   %.2f of %.2f GiB (%.1f%%), %.1f MB/s now, %.1f MB/s on average, ~%s left\n",
			float64(got)/(1<<30), float64(total)/(1<<30), 100*float64(got)/float64(total),
			float64(got-last)/now.Sub(tLast).Seconds()/1e6, rate/1e6, eta)
		last, tLast = got, now
	}
}
