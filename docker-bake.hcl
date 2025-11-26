target "dev" {
  context    = "."
  dockerfile = "Dockerfile"
  target = "dev"
  args = {
    wapiti_source_download_url = "https://github.com/kermitt2/Wapiti/archive/5f9a52351fddf21916008daa4becd41d56e7f608.tar.gz"
  }
}

target "delft" {
  context    = "."
  dockerfile = "Dockerfile"
  target = "delft"
  contexts = {
    "dev" = "target:dev"
  }
}

target "lint-flake8" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "lint-flake8"
  contexts = {
    "dev" = "target:dev"
  }
}

target "lint-pylint" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "lint-pylint"
  contexts = {
    "dev" = "target:dev"
  }
}

target "lint-mypy" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "lint-mypy"
  contexts = {
    "dev" = "target:dev"
  }
}

target "pytest-not-slow" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "pytest-not-slow"
  contexts = {
    "dev" = "target:dev"
  }
}

target "pytest-slow" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "pytest-slow"
  contexts = {
    "dev" = "target:dev"
  }
}

group "default" {
  targets = ["delft"]
}
