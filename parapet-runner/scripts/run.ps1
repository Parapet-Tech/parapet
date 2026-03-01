param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsFromCaller
)

python -m parapet_runner.runner @ArgsFromCaller
