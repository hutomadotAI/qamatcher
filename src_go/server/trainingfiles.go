package server

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/google/uuid"
	log "github.com/sirupsen/logrus"
)

type AiStatusEnum int

const (
	AiStatusEnumUndefined        AiStatusEnum = 0
	AiStatusEnumReadyToTrain     AiStatusEnum = 1
	AiStatusEnumTrainingQueued   AiStatusEnum = 2
	AiStatusEnumTraining         AiStatusEnum = 3
	AiStatusEnumTrainingStopped  AiStatusEnum = 4
	AiStatusEnumTrainingComplete AiStatusEnum = 5
	AiStatusEnumError            AiStatusEnum = 6
)

type trainingFiles struct {
	aipath string
}

func (training trainingFiles) findAllAis() error {
	log.Info(fmt.Sprintf("In findAllAis: %s", training.aipath))
	dirPath := training.aipath
	dirEntries, err := ioutil.ReadDir(dirPath)
	if err != nil {
		return err
	}

	for _, entry := range dirEntries {
		if entry.IsDir() {
			devIDString := entry.Name()
			_, err := uuid.Parse(devIDString)
			if err != nil {
				log.Info("Skipping non UUID DevID " + devIDString)
				continue
			}

			devIDPath := filepath.Join(dirPath, devIDString)
			devIDEntries, err := ioutil.ReadDir(devIDPath)
			if err != nil {
				return err
			}
			for _, entry := range devIDEntries {
				aiIDString := entry.Name()
				_, err := uuid.Parse(aiIDString)
				if err != nil {
					log.Info("Skipping non UUID AIID " + aiIDString)
					continue
				}
				aiIDPath := filepath.Join(devIDPath, aiIDString)
				trainingDataFile := filepath.Join(aiIDPath, "training_combined.txt")
				trainingStatusFile := filepath.Join(aiIDPath, "training_status.pkl")
				parseAiStatus(trainingDataFile, trainingStatusFile)
			}
		}
	}
	return nil
}

func parseAiStatus(trainingDataFile string, trainingStatusFile string) (AiStatusEnum, error) {
	if !fileExists(trainingDataFile) {
		log.Info("AI undefined as missing training data file: " + trainingDataFile)
		return AiStatusEnumUndefined, nil
	}
	if !fileExists(trainingStatusFile) {
		log.Info("AI ready to train, missing status file" + trainingStatusFile)
		return AiStatusEnumReadyToTrain, nil
	}

	file, err := os.Open(trainingStatusFile)
	if err != nil {
		log.Error(err)
		return AiStatusEnumError, nil
	}
	defer file.Close()

	//d := ogórek.NewDecoder(file)
	//obj, err := d.Decode()

	return AiStatusEnumUndefined, nil
}

func fileExists(name string) bool {
	if _, err := os.Stat(name); err != nil {
		if os.IsNotExist(err) {
			return false
		}
	}
	return true
}
