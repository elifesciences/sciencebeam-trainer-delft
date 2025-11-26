target "delft" {
  context    = "."
  dockerfile = "Dockerfile"
  target = "delft"
}

group "default" {
  targets = ["delft"]
}
