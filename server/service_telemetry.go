package server

import (
	"context"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/samyfodil/jitllm/server/gen/jitllm/v1"
)

// TelemetryService implements jitllm.v1.TelemetryService. Every number here
// is a counter the engine already keeps, never a timing this server took:
// counters are facts about what ran, not perf claims.
type TelemetryService struct{ E *Engine }

func (s *TelemetryService) GetStats(ctx context.Context, req *connect.Request[v1.GetStatsRequest]) (*connect.Response[v1.GetStatsResponse], error) {
	return connect.NewResponse(s.snapshot(req.Msg.ModelId, req.Msg.SessionId)), nil
}

func (s *TelemetryService) WatchStats(ctx context.Context, req *connect.Request[v1.WatchStatsRequest], st *connect.ServerStream[v1.GetStatsResponse]) error {
	iv := time.Duration(req.Msg.IntervalMillis) * time.Millisecond
	if iv < 250*time.Millisecond {
		iv = 250 * time.Millisecond
	}
	if err := st.Send(s.snapshot(req.Msg.ModelId, req.Msg.SessionId)); err != nil {
		return err
	}
	tick := time.NewTicker(iv)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
			if err := st.Send(s.snapshot(req.Msg.ModelId, req.Msg.SessionId)); err != nil {
				return err
			}
		}
	}
}

func (s *TelemetryService) snapshot(modelID, sessionID string) *v1.GetStatsResponse {
	out := &v1.GetStatsResponse{UnixMillis: time.Now().UnixMilli()}
	for _, lm := range s.E.Models() {
		if modelID != "" && lm.id != modelID {
			continue
		}
		out.Models = append(out.Models, &v1.ModelStats{
			ModelId:         lm.id,
			Residency:       pbResidency(lm),
			Sessions:        int32(len(lm.sessions)),
			TokensGenerated: lm.tokensGenerated.Load(),
			TokensPrefilled: lm.tokensPrefilled.Load(),
			Batch:           lm.loop.pb(),
		})
	}
	for _, sess := range s.E.Sessions() {
		if sessionID != "" && sess.id != sessionID {
			continue
		}
		if modelID != "" && sess.modelID != modelID {
			continue
		}
		ss := &v1.SessionStats{
			SessionId:       sess.id,
			ModelId:         sess.modelID,
			TokensGenerated: sess.generated.Load(),
			TokensPrefilled: sess.prefilled.Load(),
			Position:        sess.snapPos.Load(),
			Generating:      sess.running.Load(),
			QueuePosition:   sess.queuePos.Load(),
		}
		// The pool and mixture counters are written by the decode loop, so
		// they are read only when the session is idle; a busy session
		// reports zeros rather than race.
		if !sess.running.Load() && sess.mu.TryLock() {
			par, ser := sess.st.JIT().Regions()
			b, l := sess.st.MoEDispatch()
			bd, ld := sess.st.MoEDownDispatch()
			ss.Pool = &v1.PoolStats{
				ParallelRegions: par,
				InlineRegions:   ser,
				Workers:         int32(sess.st.Workers()),
			}
			ss.Mixture = &v1.MixtureStats{
				BatchedDispatch:     b,
				LoopedDispatch:      l,
				BatchedDownDispatch: bd,
				LoopedDownDispatch:  ld,
			}
			sess.mu.Unlock()
		}
		out.Sessions = append(out.Sessions, ss)
	}
	if devs, err := s.E.Devices(false); err == nil {
		for _, d := range devs {
			out.Devices = append(out.Devices, s.E.pbDevice(d))
		}
	}
	host := s.E.Host()
	out.Host = pbHost(host)
	return out
}

func (s *TelemetryService) GetServerInfo(ctx context.Context, req *connect.Request[v1.GetServerInfoRequest]) (*connect.Response[v1.GetServerInfoResponse], error) {
	var backends []string
	if devs, err := s.E.Devices(false); err == nil {
		for _, d := range devs {
			if d.Available {
				backends = append(backends, d.ID)
			}
		}
	}
	return connect.NewResponse(&v1.GetServerInfoResponse{Info: &v1.ServerInfo{
		Version:           s.E.cfg.Version,
		ModelDirectory:    s.E.ModelDir(),
		StartedUnixMillis: s.E.started.UnixMilli(),
		UptimeMillis:      millis(time.Since(s.E.started)),
		AvailableBackends: backends,
		LoadedModels:      int32(len(s.E.Models())),
		OpenSessions:      int32(len(s.E.Sessions())),
	}}), nil
}
