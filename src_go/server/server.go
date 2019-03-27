package server

import (
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

type EmbStatusServer struct {
	training trainingFiles
}

func StartServer(port int, aipath string) {
	log.WithFields(log.Fields{
		"port": port, "ai_path": aipath,
	}).Info("Starting the EMB status server")

	embMux := EmbStatusServer{
		training: trainingFiles{aipath: aipath}}
	router := mux.NewRouter()
	router.HandleFunc("/", embMux.getAis).Methods("GET")
	http.Handle("/", router)
	log.Fatal(http.ListenAndServe(fmt.Sprintf("localhost:%d", port), router))
}

func (emb EmbStatusServer) getAis(w http.ResponseWriter, req *http.Request) {
	log.Info("In root")
	emb.training.findAllAis()
}
