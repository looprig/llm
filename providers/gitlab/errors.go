package gitlab

import (
	"fmt"

	"github.com/looprig/llm"
)

type CounterSupportError = llm.CounterSupportError

type DirectAccessError struct {
	Status int
	Reason string
	Err    error
}

func (e *DirectAccessError) Error() string {
	if e.Status != 0 {
		return "gitlab: direct access: " + e.Reason + " (status " + fmt.Sprint(e.Status) + ")"
	}
	return "gitlab: direct access: " + e.Reason
}

func (e *DirectAccessError) Unwrap() error { return e.Err }
