package main

import (
	"errors"
	"fmt"

	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/contextcount"
	"github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"github.com/looprig/llm/auto"
)

func main() {
	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAIResponses,
		"https://api.openai.com/v1",
		"gpt-example",
	)

	client, err := auto.New(selected, auth.APIKey("example-key"))
	if err != nil {
		panic(err)
	}
	fmt.Printf("provider=%s client-ready=%t\n", selected.Provider, client != nil)

	counter, err := auto.NewCounter(selected, auth.APIKey("example-key"))
	if err != nil {
		panic(err)
	}
	capability := counter.CounterCapability()
	fmt.Printf(
		"counter-provider=%s exact=%t separate-endpoint=%t\n",
		capability.Provider,
		capability.Quality == contextcount.CountQualityExactProvider,
		capability.Transport == contextcount.CounterTransportSeparateEndpoint,
	)

	unsupported := model.CustomModel(
		model.ProviderName(llm.ProviderXAI),
		model.APIFormatOpenAIResponses,
		"https://api.x.ai/v1",
		"grok-example",
	)
	_, err = auto.NewCounter(unsupported, auth.APIKey("example-key"))
	var supportErr *llm.CounterSupportError
	fmt.Printf("xai-exact-counter=%t\n", !errors.As(err, &supportErr))
}
