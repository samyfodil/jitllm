package app

import (
	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/session"
)

// The settings live in common/config, shared with the terminal; these are
// re-exported so the window keeps one import.
type (
	Env    = config.Env
	Config = config.Config
)

// UseEnv sets what the app takes from the environment; see [config.UseEnv].
func UseEnv(e Env) { config.UseEnv(e) }

// DefaultModelDir is re-exported; see [config.DefaultModelDir].
func DefaultModelDir() string { return config.DefaultModelDir() }

// DefaultConfig is re-exported; see [config.DefaultConfig].
func DefaultConfig() *Config { return config.DefaultConfig() }

// ConfigPath is re-exported; see [config.ConfigPath].
func ConfigPath() string { return config.ConfigPath() }

// LoadConfig is re-exported; see [config.LoadConfig].
func LoadConfig() *Config { return config.LoadConfig() }

// KVCacheDir is re-exported; see [session.KVCacheDir].
func KVCacheDir() string { return session.KVCacheDir() }
