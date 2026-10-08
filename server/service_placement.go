package server

import (
	"context"
	"fmt"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/samyfodil/jitllm/server/gen/jitllm/v1"
)

// PlacementService implements jitllm.v1.PlacementService: the engine's block
// placement, seam, relocation, seam tuning and page budget, reachable from
// outside the process.
type PlacementService struct{ E *Engine }

func (s *PlacementService) GetPlacement(ctx context.Context, req *connect.Request[v1.GetPlacementRequest]) (*connect.Response[v1.GetPlacementResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.GetPlacementResponse{Placement: s.E.pbPlacement(sess)}), nil
}

// SetPlacement moves the seam. It takes the session's lock, so it waits for
// a generate to finish rather than mutating the State mid-Forward.
func (s *PlacementService) SetPlacement(ctx context.Context, req *connect.Request[v1.SetPlacementRequest]) (*connect.Response[v1.SetPlacementResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	m := req.Msg
	want := normaliseDeviceIDs(m.DeviceIds)
	if len(want) > 0 && sess.lm.dev == nil {
		return nil, connect.NewError(connect.CodeFailedPrecondition, fmt.Errorf(
			"session %q was loaded host-only: its model holds no tier, so there is no device "+
				"to offer blocks to. Load the model with device_ids to make placement reachable",
			sess.id))
	}
	if len(want) > 0 && !sameSet(want, sess.lm.deviceIDs) {
		// The tier is opened at LOAD and owns the model; a session cannot
		// re-choose the hardware under it.
		return nil, connect.NewError(connect.CodeFailedPrecondition, fmt.Errorf(
			"session %q is on tier %v and cannot be moved to %v: the TIER owns the model and is "+
				"opened at load. Load the model again with the devices you want",
			sess.id, sess.lm.deviceIDs, want))
	}

	start := time.Now()
	sess.mu.Lock()
	if len(want) == 0 {
		// Everything home.
		sess.st.SetGPULayers(0)
	} else {
		max := -1
		if m.MaxDeviceBlocks != nil {
			max = int(*m.MaxDeviceBlocks)
		}
		if max < 0 {
			if err := sess.st.SetDeviceLayers(sess.lm.dev, -1); err != nil {
				sess.mu.Unlock()
				return nil, connect.NewError(connect.CodeFailedPrecondition, err)
			}
		} else {
			sess.st.SetGPULayers(max)
		}
	}
	if m.HeadOnDevice != nil {
		sess.st.SetHeadOnDevice(*m.HeadOnDevice)
	}
	sess.refresh()
	sess.mu.Unlock()

	return connect.NewResponse(&v1.SetPlacementResponse{
		Placement: s.E.pbPlacement(sess),
		Millis:    millis(time.Since(start)),
	}), nil
}

// RelocateBlocks moves a contiguous block range between the host and the
// device while serving. The engine's seam is a count (SetGPULayers(n): the
// first n blocks are on the device), so a range that extends or trims the
// device set is honoured. These are refused with their reason:
//
//	a hole in the middle   the seam is a count; there is no "blocks 3..7 on
//	                       the card and 8..12 at home" to set.
//	device -> device       the tier chooses which of its devices takes a
//	                       block (fastest first) and exposes only the
//	                       per-device COUNT. Naming a destination device per
//	                       block would be a guess dressed as an API.
//	a host-only model      there is no tier to move to.
func (s *PlacementService) RelocateBlocks(ctx context.Context, req *connect.Request[v1.RelocateBlocksRequest]) (*connect.Response[v1.RelocateBlocksResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	m := req.Msg
	nLayer := sess.lm.m.Cfg.NLayer
	first, last := int(m.FirstBlock), int(m.LastBlock)
	if first < 0 || last < first || last >= nLayer {
		return nil, invalid("relocate: block range %d..%d is outside 0..%d", first, last, nLayer-1)
	}
	if !m.ToHost && m.ToDeviceId == "" {
		return nil, invalid("relocate: set to_host, or name a to_device_id")
	}
	if m.ToDeviceId != "" && sess.lm.dev == nil {
		return nil, connect.NewError(connect.CodeFailedPrecondition, fmt.Errorf(
			"session %q was loaded host-only; there is no device to move blocks to", sess.id))
	}
	if m.ToDeviceId != "" && !hasString(sess.lm.deviceIDs, m.ToDeviceId) {
		return nil, unimplemented(
			"relocate: this session's tier holds %v, and moving a block to %q would be a "+
				"device-to-device move. The tier chooses which of its devices takes a block "+
				"and exposes only the per-device count, so there is no per-block destination "+
				"to set -- this is declined rather than approximated",
			sess.lm.deviceIDs, m.ToDeviceId)
	}

	start := time.Now()
	sess.mu.Lock()
	cur := sess.st.GPULayers()
	var moved int
	var declined []*v1.PlacementDecline
	switch {
	case m.ToHost:
		if last != cur-1 {
			sess.mu.Unlock()
			return nil, unimplemented(
				"relocate: the seam is a COUNT -- blocks 0..%d are on the device -- so bringing "+
					"%d..%d home would leave a hole. Only the TAIL of the device set can be "+
					"brought home; ask for %d..%d",
				cur-1, first, last, first, cur-1)
		}
		got := sess.st.SetGPULayers(first)
		moved = cur - got
	default:
		if first != cur {
			sess.mu.Unlock()
			return nil, unimplemented(
				"relocate: the seam is a COUNT -- blocks 0..%d are on the device -- so placing "+
					"%d..%d would leave a hole. Only a range that EXTENDS the device set can be "+
					"placed; ask for %d..%d",
				cur-1, first, last, cur, last)
		}
		got := sess.st.SetGPULayers(last + 1)
		moved = got - cur
		if got < last+1 {
			declined = append(declined, &v1.PlacementDecline{
				BlockIndex:  int32(got),
				DeviceId:    m.ToDeviceId,
				Reason:      fmt.Sprintf("the device took %d of the %d block(s) asked for", moved, last+1-cur),
				OutOfMemory: true,
			})
		}
	}
	sess.refresh()
	sess.mu.Unlock()

	return connect.NewResponse(&v1.RelocateBlocksResponse{
		Placement:   s.E.pbPlacement(sess),
		BlocksMoved: int32(moved),
		Declined:    declined,
		Millis:      millis(time.Since(start)),
	}), nil
}

func (s *PlacementService) SetRelocation(ctx context.Context, req *connect.Request[v1.SetRelocationRequest]) (*connect.Response[v1.SetRelocationResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	sess.mu.Lock()
	sess.st.SetRelocate(req.Msg.Enabled)
	sess.refresh()
	sess.mu.Unlock()
	return connect.NewResponse(&v1.SetRelocationResponse{Enabled: req.Msg.Enabled}), nil
}

// TuneSeam arms the engine's own measured re-placement and reports what it
// settles on. seamtune measures over runs of real tokens, so it only
// advances while the session is generating; this call arms it and reports its
// state without decoding throwaway tokens.
func (s *PlacementService) TuneSeam(ctx context.Context, req *connect.Request[v1.TuneSeamRequest], st *connect.ServerStream[v1.TuneSeamResponse]) error {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return connectErr(err)
	}
	if sess.lm.dev == nil {
		return connect.NewError(connect.CodeFailedPrecondition, fmt.Errorf(
			"session %q is host-only: there is no seam to tune", sess.id))
	}
	sess.mu.Lock()
	sess.st.SetSeamTuning(true)
	m := req.Msg
	sess.st.SetSeamSchedule(int(m.WarmupTokens), int(m.TokensPerRun), int(m.Rounds))
	sess.refresh()
	sess.mu.Unlock()

	if err := st.Send(&v1.TuneSeamResponse{
		Round: 0,
		Note: "the tuner is armed. It measures over RUNS OF TOKENS and has none of its own, " +
			"so it advances only while this session is generating; keep this stream open and " +
			"send generate requests on the session.",
	}); err != nil {
		return err
	}

	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	round := int32(0)
	lastBlocks := -1
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
			// The snapshot, not the State: Forward writes the tuner with no
			// lock, so it is read as of the session's last refresh.
			blocks, settled := int(sess.snapSeam.Load()), sess.snapSeamSettled.Load()
			if blocks == lastBlocks && !settled {
				continue
			}
			lastBlocks = blocks
			round++
			resp := &v1.TuneSeamResponse{
				Round:                 round,
				CandidateDeviceBlocks: int32(blocks),
				Accepted:              settled,
				Done:                  settled,
				Placement:             s.E.pbPlacement(sess),
			}
			if settled {
				resp.Note = "settled"
			}
			if err := st.Send(resp); err != nil {
				return err
			}
			if settled {
				return nil
			}
		}
	}
}

func (s *PlacementService) GetResidency(ctx context.Context, req *connect.Request[v1.GetResidencyRequest]) (*connect.Response[v1.GetResidencyResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.GetResidencyResponse{Residency: pbResidency(lm)}), nil
}

// SetPageBudget retargets the pager. The response carries fits because a
// budget one page short is a cliff, not a gradient (see pbResidency).
func (s *PlacementService) SetPageBudget(ctx context.Context, req *connect.Request[v1.SetPageBudgetRequest]) (*connect.Response[v1.SetPageBudgetResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	// The request names a total and pins it: the other models divide what is
	// left (Engine.Pin), and every session created afterwards re-divides it
	// (Engine.applyPageBudget). What the pager gets is that total less the
	// never-paged dense region and less what a tier on the host's own memory
	// holds (Model.SetPageBudget); Residency.budget reports what was actually
	// applied, so the response is never a repeat of the request. Zero releases
	// the pin back to the engine's division of the host budget.
	if err := s.E.Pin(lm.id, req.Msg.BudgetBytes); err != nil {
		return nil, connectErr(err)
	}
	res := pbResidency(lm)
	return connect.NewResponse(&v1.SetPageBudgetResponse{Residency: res, Fits: res.Fits}), nil
}

func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for _, x := range a {
		if !hasString(b, x) {
			return false
		}
	}
	return true
}
