#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# ================================
# Configuration - Customize these!
# ================================
PROJECT_ID="wharfside-govdocs"            # Replace with your Google Cloud project ID.
BUCKET_NAME="govdocs-wharfside"     # Must be globally unique. Use your domain if applicable.
REGION="us-east4"                      # Bucket region; adjust if needed.
WEBSITE_DIR="./website7"                   # Local directory containing your website files.
INDEX_FILE="index.html"                   # Your homepage file.
ERROR_FILE="404.html"                     # Optional error page; update if you have one.

# ================================
# Step 1: Set the active project.
# ================================
echo "Setting the active project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# ================================
# Step 2: Create the Cloud Storage bucket.
# ================================
#echo "Creating bucket: gs://$BUCKET_NAME in region $REGION..."
#gsutil mb -l "$REGION" gs://"$BUCKET_NAME"

# ================================
# Step 3: Upload website files to the bucket.
# ================================
#echo "Uploading website files from $WEBSITE_DIR to gs://$BUCKET_NAME..."
#gsutil -m rsync -R "$WEBSITE_DIR" gs://"$BUCKET_NAME"

# ================================
# Step 4: Configure the bucket for static website hosting.
# ================================
echo "Configuring bucket for website hosting..."
gsutil web set -m "$INDEX_FILE" -e "$ERROR_FILE" gs://"$BUCKET_NAME"

# ================================
# Step 5: Make the bucket objects publicly accessible.
# ================================
echo "Setting public read permissions on bucket objects..."
gsutil iam ch allUsers:objectViewer gs://"$BUCKET_NAME"

# ================================
# Completion Message
# ================================
echo "Static website deployed successfully!"
echo "You can access your site at: http://storage.googleapis.com/$BUCKET_NAME/$INDEX_FILE"
