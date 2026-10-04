FROM ubuntu:26.04

ENV RUNNER_VERSION=2.334.0

LABEL RunnerVersion=${RUNNER_VERSION}
LABEL org.opencontainers.image.source="https://github.com/aladin-heig/mlops-guide"

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y build-essential lsb-release python3 python3-pip \
    curl jq vim gpg git unzip tar gettext-base libicu-dev

WORKDIR /home/ubuntu

# Install GitHub Actions Runner
RUN curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
RUN tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Install extra dependencies for the runner
RUN ./bin/installdependencies.sh

COPY startup.sh .

RUN chmod +x startup.sh

# Ensure everything in the workdir belongs to the ubuntu user.
RUN chown -R ubuntu:ubuntu /home/ubuntu && \
    chmod -R u+rwX /home/ubuntu

USER ubuntu

ENTRYPOINT ["./startup.sh"]
