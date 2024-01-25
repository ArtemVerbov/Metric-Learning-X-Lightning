.PHONY: *

GOOGLE_DRIVE_DATASET := .google_drive_dataset

CLEARML_PROJECT_NAME := metric_learning
CLEARML_DATASET_NAME := metric_learning_dataset


migrate_dataset:
	# Migrate dataset to ClearML datasets.
	rm -R $(GOOGLE_DRIVE_DATASET) || true
	mkdir $(GOOGLE_DRIVE_DATASET)
	wget "https://drive.usercontent.google.com/download?id=1IGa__ZsO6vxzpfbCn0A1SxK_WGKe4a-9&export=download&confirm=yes" -O $(GOOGLE_DRIVE_DATASET)/dataset.zip
	unzip -q $(GOOGLE_DRIVE_DATASET)/dataset.zip -d $(GOOGLE_DRIVE_DATASET) || true
	rm $(GOOGLE_DRIVE_DATASET)/dataset.zip
	find $(GOOGLE_DRIVE_DATASET) -type f -name '.DS_Store' -delete

	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	clearml-data add --files $(GOOGLE_DRIVE_DATASET)
	clearml-data close --verbose
	rm -R $(GOOGLE_DRIVE_DATASET)

