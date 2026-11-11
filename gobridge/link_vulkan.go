//go:build vulkan

package moe

/*
#cgo LDFLAGS: -L${SRCDIR}/../src/moe_stream/build -lmoe_bridge -lvulkan -lm -lpthread
*/
import "C"
