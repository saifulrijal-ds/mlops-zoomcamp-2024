FROM mageai/mageai:alpha

ARG USER_CODE_PATH=/home/src/${PROJECT_NAME}

# Note: this overwrites the requirements.txt file in your new project on first run. 
# You can delete this line for the second run :) 
COPY requirements.txt ${USER_CODE_PATH}/requirements.txt 

RUN pip3 install -r ${USER_CODE_PATH}/requirements.txt

# Copy the startup script
COPY start-mage.sh /app/start-mage.sh

# Ensure the startup scripts is executable
RUN chmod +x /app/start-mage.sh

# CMD ["/bin/sh", "-c", "/app/run_app.sh"]
# Set the default entrypoint command
ENTRYPOINT [ "/bin/sh", "-c", "/app/start-mage.sh" ]