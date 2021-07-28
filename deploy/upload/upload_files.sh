#!/bin/sh

UPLOAD_FOLDER=$APP_NAME
mkdir -p $UPLOAD_FOLDER/$CHART_VERSION

# generate new version file
{
	echo "--Public:"
	echo "CHART_NAME: $CHART_NAME"
    echo "CHART_VERSION: $CHART_VERSION"
    echo "CHART_DESCRIPTION: $CHART_DESCRIPTION"
    echo "IMAGE_LOCATION: $IMAGE_REPOS:$IMAGE_TAG"
    echo -e "\n--Internal:"
    echo "PROJECT_URL: $CI_PROJECT_URL"
    echo "COMMIT_BRANCH: $CI_COMMIT_REF_SLUG"
    echo "COMMIT_HASH_REF: $CI_REF"

} > ./$UPLOAD_FOLDER/$CHART_VERSION/RELEASE_NOTES.txt


# export k8s yaml files
cd ../k8s
sh export_k8s_yaml_all.sh
cd ../upload


#===== **Modify Accordingly**: Prepare All Files To Upload =====

# get README
rsync -av ../../README.md ./$UPLOAD_FOLDER/$CHART_VERSION/

# get postman collection files
rsync -av ../test/collections ./$UPLOAD_FOLDER/$CHART_VERSION/test/

# get k8s all-in-one yaml file
rsync -av ../k8s/deploy-k8s.yaml ./$UPLOAD_FOLDER/$CHART_VERSION/k8s/

# get requirement and values yaml file if exist
rsync -av ../k8s/$CHART_NAME/values.yaml ./$UPLOAD_FOLDER/$CHART_VERSION/k8s/chart/ || true
rsync -av ../k8s/$CHART_NAME/requirements.yaml ./$UPLOAD_FOLDER/$CHART_VERSION/k8s/chart/ || true

# get all properties
rsync -av ../k8s/$CHART_NAME/configs ./$UPLOAD_FOLDER/$CHART_VERSION/k8s/

#========================================



#### NO NEED TO CHANGE BELOW ####

SPEC_LOGIN="spec:spec0000"

create_remote_folder()
{
	BASE_URL=$1
	FOLDER=$2

	# check if folder exists, otherwise create one
	#if [[ $(curl -X PROPFIND -H "Depth: $DEPTH" -u $SPEC_LOGIN "$BASE_URL" | grep "/$FOLDER") ]]; then
	if curl -u $SPEC_LOGIN --output /dev/null --silent --head --fail "$BASE_URL/$FOLDER"; then
		echo "Folder $FOLDER already exists"
	else
		echo "Folder $FOLDER not found, create it"
		curl -u $SPEC_LOGIN -X MKCOL "$BASE_URL/$FOLDER"
	fi
}


# loop through upload folder to upload files/directories
upload_files()
{
	BASE_URL=$1
	FOLDER=$2
	DEPTH=1
	for file in $FOLDER/*; do

		if ! [ "$(ls -A $file)" ]; then continue; fi

		if [[ -d "$file" && ! -L "$file" ]]; then
			echo "$file is a directory, create remote directory";

			foldername=$(basename "$file")

			create_remote_folder $BASE_URL $file

			upload_files $BASE_URL $file

		else
			echo "found file with name: $file"

			filename=$(basename "$file")
			curl -u $SPEC_LOGIN -X DELETE "$BASE_URL/$file"
			curl -u $SPEC_LOGIN -T $file "$BASE_URL/$file"
		fi

	done
}


OWN_CLOUD_SPEC_URL="http://ess-repos01.wisers.com:10000/remote.php/dav/files/spec/Spec"

create_remote_folder $OWN_CLOUD_SPEC_URL $UPLOAD_FOLDER
upload_files $OWN_CLOUD_SPEC_URL $UPLOAD_FOLDER
