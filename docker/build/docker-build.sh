#!/bin/bash
# This is directly from https://gitlab.com/ska-telescope/ska-cicd-k8s-tools/-/blob/master/images/ska-cicd-k8s-tools-build-deploy/scripts/docker-build.sh
# A rework of how this project is handled in CI is required but out of scope for time allocated to fixing metadata.

label () {
  LINE=${*}
  KEY=${LINE%=*}
  VALUE=${LINE#*=}
  if [ -z "$LABELS" ]
  then
  LABELS="--label $KEY=\"$VALUE\""
  else
  LABELS="${LABELS} --label $KEY=\"$VALUE\""
  fi
  echo "--label $KEY=\"$VALUE\""
}

while IFS='' read -r LINE || [ -n "${LINE}" ]; do
    if [[ $LINE == *"CI_COMMIT_AUTHOR"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_COMMIT_REF_NAME"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_COMMIT_REF_SLUG"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_COMMIT_SHA"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_COMMIT_SHORT_SHA"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_COMMIT_TIMESTAMP"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_JOB_ID"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_JOB_URL"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_PIPELINE_ID"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_PIPELINE_IID"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_PIPELINE_URL"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_PROJECT_ID"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_PROJECT_PATH_SLUG"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_PROJECT_URL"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_RUNNER_ID"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_RUNNER_REVISION"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"CI_RUNNER_TAGS"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"GITLAB_USER_NAME"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"GITLAB_USER_EMAIL"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"GITLAB_USER_LOGIN"* ]]; then
        label $LINE
    fi
    if [[ $LINE == *"GITLAB_USER_ID"* ]]; then
        label $LINE
    fi
done <<< "$(printenv)"

DOCKER_BUILD_COMMAND="docker build $LABELS -t $1 $2"
echo "$DOCKER_BUILD_COMMAND"
bash -c "$DOCKER_BUILD_COMMAND"
