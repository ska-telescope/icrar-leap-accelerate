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

echo docker build $LABELS -t $1 $2
#docker build $LABELS -t $1 $2
docker build --label GITLAB_USER_ID="4206580" --label GITLAB_USER_EMAIL="mark@boulton.net" --label GITLAB_USER_LOGIN="markaboulton" --label CI_PIPELINE_ID="529970707" --label CI_COMMIT_REF_SLUG="yan-890" --label CI_COMMIT_SHORT_SHA="c87869bf" --label CI_RUNNER_TAGS="k8srunner" --label CI_COMMIT_SHA="c87869bf119ffc8efffe68aa2129c3e545fb596f" --label CI_PIPELINE_IID="1017" --label CI_PIPELINE_URL="https://gitlab.com/ska-telescope/icrar-leap-accelerate/-/pipelines/529970707" --label GITLAB_USER_NAME="markaboulton" --label CI_RUNNER_ID="14458924" --label CI_COMMIT_TIMESTAMP="2022-05-03T17:05:25+08:00" --label CI_COMMIT_AUTHOR="Mark Boulton <mark@boulton.net>" --label CI_COMMIT_REF_NAME="YAN-890" --label CI_JOB_URL="https://gitlab.com/ska-telescope/icrar-leap-accelerate/-/jobs/2406478686" --label CI_RUNNER_REVISION="5316d4ac" --label CI_PROJECT_PATH_SLUG="ska-telescope-icrar-leap-accelerate" --label CI_JOB_ID="2406478686" --label CI_PROJECT_ID="22944835" --label CI_PROJECT_URL="https://gitlab.com/ska-telescope/icrar-leap-accelerate" -t $1 $2
