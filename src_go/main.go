package main

import (
	"flag"
	"fmt"
	"hutomadotai/embstatus/server"
	"os"
	"time"

	log "github.com/sirupsen/logrus"
)

var port = flag.Int("port", 8000, "Server port")
var aipath = flag.String("aipath", "/ai", "AI file path")

func main() {
	flag.Parse()
	log.SetFormatter(&log.JSONFormatter{TimestampFormat: time.RFC3339Nano})
	log.SetOutput(os.Stdout)
	log.SetLevel(log.InfoLevel)
	log.Info(fmt.Sprintf(
		"Starting server with port %d, local path %s", *port, *aipath))
	server.StartServer(*port, *aipath)
}
