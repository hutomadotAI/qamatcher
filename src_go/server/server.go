package server

import (
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

type EmbStatusServer struct {
	aipath string
}

func StartServer(port int, aipath string) {
	log.WithFields(log.Fields{
		"port": port, "ai_path": aipath,
	}).Info("Starting the server")

	embMux := EmbStatusServer{aipath: aipath}
	router := mux.NewRouter()
	router.HandleFunc("/", embMux.getAis).Methods("GET")
	http.Handle("/", router)
	http.ListenAndServe(fmt.Sprintf("localhost:%d", port), router)
}

func (emb EmbStatusServer) getAis(w http.ResponseWriter, req *http.Request) {
	log.Info("In root")
}
