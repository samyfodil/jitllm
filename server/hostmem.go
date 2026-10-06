package server

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"sync"
)

// hostTotal and hostAvailable read /proc/meminfo directly rather than adding a
// dependency. On a host with no /proc they return 0, which callers treat as
// "not reported"; nothing divides by these.
var meminfoOnce struct {
	sync.Mutex
	read  bool
	total uint64
}

func hostTotal() uint64 {
	meminfoOnce.Lock()
	defer meminfoOnce.Unlock()
	if meminfoOnce.read {
		return meminfoOnce.total
	}
	meminfoOnce.read = true
	meminfoOnce.total = meminfoField("MemTotal:")
	return meminfoOnce.total
}

// hostAvailable is not cached: it is the quantity that moves.
func hostAvailable() uint64 { return meminfoField("MemAvailable:") }

func meminfoField(key string) uint64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, key) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}
