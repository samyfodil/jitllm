package server

import (
	"context"
	"fmt"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// DeviceService implements jitllm.v1.DeviceService.
type DeviceService struct{ E *Engine }

func (s *DeviceService) ListDevices(ctx context.Context, req *connect.Request[v1.ListDevicesRequest]) (*connect.Response[v1.ListDevicesResponse], error) {
	devs, err := s.E.Devices(false)
	if err != nil {
		return nil, connectErr(err)
	}
	out := &v1.ListDevicesResponse{}
	for _, d := range devs {
		out.Devices = append(out.Devices, s.E.pbDevice(d))
	}
	return connect.NewResponse(out), nil
}

func (s *DeviceService) GetDevice(ctx context.Context, req *connect.Request[v1.GetDeviceRequest]) (*connect.Response[v1.GetDeviceResponse], error) {
	devs, err := s.E.Devices(false)
	if err != nil {
		return nil, connectErr(err)
	}
	for _, d := range devs {
		if d.ID == req.Msg.DeviceId {
			return connect.NewResponse(&v1.GetDeviceResponse{Device: s.E.pbDevice(d)}), nil
		}
	}
	return nil, connect.NewError(connect.CodeNotFound,
		fmt.Errorf("no device %q; call ListDevices for the ids this host answers to", req.Msg.DeviceId))
}

// GetMemoryTopology answers "how many bytes can I spend". spendable_total is
// the host's weight budget plus only the devices whose memory is their own
// (see SpendableTotal), so no client double-counts an integrated GPU.
func (s *DeviceService) GetMemoryTopology(ctx context.Context, req *connect.Request[v1.GetMemoryTopologyRequest]) (*connect.Response[v1.GetMemoryTopologyResponse], error) {
	devs, err := s.E.Devices(false)
	if err != nil {
		return nil, connectErr(err)
	}
	host := s.E.Host()
	top := &v1.MemoryTopology{
		Host:           pbHost(host),
		SpendableTotal: pbBytes(SpendableTotal(host, devs)),
	}
	for _, d := range devs {
		top.Devices = append(top.Devices, s.E.pbDevice(d))
	}
	return connect.NewResponse(&v1.GetMemoryTopologyResponse{Topology: top}), nil
}
