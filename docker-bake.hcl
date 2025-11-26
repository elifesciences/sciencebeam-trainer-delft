target "delft" {
  context    = "."
  dockerfile = "Dockerfile"
}

group "default" {
  targets = ["delft"]
}
