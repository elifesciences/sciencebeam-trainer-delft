target "delft" {
  context    = "."
  dockerfile = "Dockerfile"
  target = "delft"
}

target "lint-flake8" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "lint-flake8"
}

target "lint-pylint" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "lint-pylint"
}

target "lint-mypy" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "lint-mypy"
}

target "pytest-not-slow" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "pytest-not-slow"
}

target "pytest-slow" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "pytest-slow"
}

group "default" {
  targets = ["delft"]
}
