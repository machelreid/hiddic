#!/bin/bash

DRY=$1
SOURCE_DIR='/home/machelreid/hiddic/'
TARGET_DIR='bigguy:/home/machel.reid/hiddic/'

if [ "$HOSTNAME" = 'shikamaru' ]; then
    echo "Copying from $SOURCE_DIR to $TARGET_DIR"
    # a:archive mode r: recursive v: verbose
    if [ "$DRY" = n ]; then
        rsync -arvn --update --exclude-from=.rsync_ignore $SOURCE_DIR $TARGET_DIR
        #printf '%s\n' "dry run"
    else
        rsync -arv --update --exclude-from=.rsync_ignore $SOURCE_DIR $TARGET_DIR
        #printf '%s\n' "wet run"
    fi
else
    printf '%s\n' "Script not executed from shikamaru. Please switch to shikamaru and rerun this script."
fi


