pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "sreejachalla/wine-quality-api"
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds')
    }

    stages {

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m pip install --upgrade pip --break-system-packages'
                sh 'pip3 install -r requirements.txt --break-system-packages'
            }
        }

        stage('Train Model & Print Metrics') {
            steps {
                sh '''
                echo "=================================="
                echo "Name: CHALLA SREEJA"
                echo "Roll No: 2022BCS0121"
                echo "=================================="
                python3 train.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_IMAGE .'
            }
        }

        stage('Push to DockerHub') {
            steps {
                sh '''
                echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin
                docker push $DOCKER_IMAGE
                '''
            }
        }
    }
}
