# L3 Outbound: Tool Call Constraints

L3 outbound validates model-emitted tool calls against contract policy before tool execution.

It enforces allowlists and argument constraints using a deterministic 9-predicate DSL.

## Config

```yaml
tools:
  _default:
    allowed: false
  web_search:
    allowed: true
    constraints:
      query:
        type: string
        max_length: 500
      url:
        type: string
        url_host: ["example.com"]

layers:
  L3_outbound:
    mode: block
    block_action: rewrite   # rewrite | error
```

## Constraint predicates

For each constrained argument, all configured predicates must pass:

1. `type`
2. `starts_with`
3. `not_contains`
4. `matches`
5. `one_of`
6. `max_length`
7. `min`
8. `max`
9. `url_host`

Missing constrained arguments are blocked.

String comparisons are normalized for bypass resistance.

## Modes and actions

`L3_outbound.mode`:

1. `block`: enforce tool constraints.
2. non-`block`: allow all tool calls.

`block_action` when blocked:

1. `error`: return `403`.
2. `rewrite` (default): remove blocked tool calls and inject refusal text into response.

This behavior is applied in both non-streaming and streaming output paths.
