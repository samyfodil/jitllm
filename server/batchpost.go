package server

import "sync/atomic"

// stepPost is the step loop's helper goroutine: while a step runs on the
// device or the pool, it turns the tokens the step feeds into text and
// events (streamText.push, row.push), the Go work a token costs that the
// step does not depend on.
//
// What moved off the critical path, by construction: each token's text
// (streamText.push, which decodes only the ids new since the last push),
// the event's allocation and its append to the row's outbox, and the wake
// of the request goroutine. What stays on it: sampling (the step needs the token)
// and, for a row with stop strings, the text, since a stop string decides
// whether the token is fed at all.
//
// The loop adds jobs before the step (add), hands them over (start) and
// waits for them after it (wait); a job touches only its row's stream and
// outbox, which nothing else touches until wait returns. Every path that
// retires a row runs after wait, so a row's last token event precedes its
// flushed tail and its done.
type stepPost struct {
	jobs  []postJob
	busy  bool
	work  chan []postJob
	ended chan int
	// posted counts the jobs done (BatchStats' accounting, batchCounters).
	posted *atomic.Int64
}

// postJob is one sampled token of one row: its id and its index in the
// completion.
type postJob struct {
	r     *row
	id    int32
	index int
	lp    *TokenLogprob
}

func newStepPost(posted *atomic.Int64) *stepPost {
	p := &stepPost{work: make(chan []postJob), ended: make(chan int), posted: posted}
	go p.serve()
	return p
}

func (p *stepPost) serve() {
	for jobs := range p.work {
		for _, j := range jobs {
			r := j.r
			chunk, _, _ := r.stream.push(r.out[:j.index+1])
			r.push(Event{Kind: EventToken, Token: &Token{ID: j.id, Text: chunk, Index: j.index, Logprob: j.lp}})
		}
		p.ended <- len(jobs)
	}
}

// add queues a token of r for the next start. r.out holds it already.
func (p *stepPost) add(r *row, id int32, index int, lp *TokenLogprob) {
	p.jobs = append(p.jobs, postJob{r, id, index, lp})
}

// start hands the queued jobs to the helper.
func (p *stepPost) start() {
	if len(p.jobs) == 0 {
		return
	}
	p.busy = true
	p.work <- p.jobs
}

// wait blocks until the started jobs are done; with none running it returns
// at once, so a path that retires a row may call it whenever.
func (p *stepPost) wait() {
	if !p.busy {
		return
	}
	p.posted.Add(int64(<-p.ended))
	p.busy = false
	clear(p.jobs)
	p.jobs = p.jobs[:0]
}

// close ends the helper. The loop has stopped, so nothing is started.
func (p *stepPost) close() { close(p.work) }
