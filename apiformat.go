package llm

import "github.com/looprig/inference"

// APIFormatBedrockConverse is the provider-named wire dialect for AWS Bedrock's
// native Converse API. The OpenAI/Anthropic/Gemini dialect names live in inference
// (inference.APIFormatOpenAI/Anthropic/Gemini); this one stays in llm because the
// Converse codec, region routing, and SigV4 wiring are provider policy that has no
// home in the neutral module yet. It is an ordinary inference.APIFormat value: the
// open label type carries no built-in validation gate.
const APIFormatBedrockConverse inference.APIFormat = "bedrock-converse"
