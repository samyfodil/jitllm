package main

import (
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

// planConvert is `jitllm convert -plan`: the whole conversion of a safetensors
// model run up to its first tensor byte (convert.PlanShards), reading the
// shards' headers only -- a streamed repository by range, nothing stored --
// and the container it would write described. A model that plans cleanly
// cannot be refused an hour into its conversion on anything a header says.
func planConvert(sm *streamed, src, dst string, q8 bool, opts []convert.Option) error {
	plan := func(o []convert.Option) (*convert.ShardPlan, error) {
		if sm != nil {
			return convert.PlanShards(sm.meta, sm.name, sm.shards, o...)
		}
		return convert.PlanSafetensors(src, o...)
	}
	var srcBytes int64
	if sm != nil {
		srcBytes = sm.size()
	} else if n, err := sourceBytes(src); err == nil {
		srcBytes = n
	}
	p, err := plan(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "plan     %s would be REFUSED: %v\n", src, err)
		return err
	}
	gb := func(n uint64) string { return fmt.Sprintf("%.2f GB (%.2f GiB)", float64(n)/1e9, float64(n)/(1<<30)) }
	h := p.Header
	fmt.Fprintf(os.Stderr, "plan     %s -> %s: nothing refused\n", src, dst)
	fmt.Fprintf(os.Stderr, "         source %s; container %s, %d bytes\n", gb(uint64(srcBytes)), gb(p.Size), p.Size)
	fmt.Fprintf(os.Stderr, "         %d tensors; %d blocks of %.2f MiB", h.NTensors, h.NBlocks,
		float64(h.PageSize)/(1<<20))
	if h.NExpPages > 0 {
		fmt.Fprintf(os.Stderr, ", %d expert pages of %.2f MiB", h.NExpPages, float64(h.ExpPageSize)/(1<<20))
	}
	fmt.Fprintf(os.Stderr, "; %d source tensor(s) named and not carried\n", p.Ignored)
	types := make([]jlm.Type, 0, len(p.ByType))
	for t := range p.ByType {
		types = append(types, t)
	}
	slices.Sort(types)
	var by []string
	var tensors uint64
	for _, t := range types {
		by = append(by, fmt.Sprintf("%v %s", t, gb(p.ByType[t])))
		tensors += p.ByType[t]
	}
	fmt.Fprintf(os.Stderr, "         by type: %s; the rest, %s, is page padding (every block page "+
		"is the largest block's)\n", strings.Join(by, ", "), gb(p.Size-min(tensors, p.Size)))
	if p.BF16 > 0 && !q8 {
		// The default carries BF16 as F32 (exact, twice the bytes); say what
		// that costs and what the two alternatives would write.
		alt, err := plan(append(slices.Clone(opts), convert.WithQ8()))
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "         BF16 in the source: %s, written as F32: %s; a BF16-preserving "+
			"store would be under %s (its pages shrink too); -q8 writes %s\n",
			gb(p.BF16), gb(2*p.BF16), gb(p.Size-p.BF16), gb(alt.Size))
	}
	return nil
}
