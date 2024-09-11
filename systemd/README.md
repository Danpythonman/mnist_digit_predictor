Setting Up systemd Service for the MNIST Digit Predictor
========================================================

The infrastructure of this site uses a combination of Docker and systemd.

- Docker containerizes the entire app and isolates it from the rest of the
  computer.

- Systemd is used to keep the Docker container running as a service and restart
  it if the computer restarts.

Note that:

- In the [docker-compose.yaml](../docker-compose.yaml) file, the
  `restart: always` option makes sure the container is always running, even in
  the event of the container failing.

- In the [systemd unit file](./mnist_digit_predictor.service), the
  `Restart=always` option makes sure the Docker container is always activated
  with the `docker compose` command, even in the event of a wider system failure
  causing a system restart.

Thus, the website is protected from both small container-wide failures and
larger system-wide failures.

What to Edit in the Systemd Unit File
-------------------------------------

The systemd unit file at
[./mnist_digit_predictor.service](./mnist_digit_predictor.service)
is a template and needs some editing before being deployed.

- Line 7:

  ```
  WorkingDirectory=/path/to/mnist_digit_predictor
  ```

  Change `/path/to/mnist_digit_predictor` to the location of the root of the
  website (where the [Dockerfile](../Dockerfile) and the
  [docker-compose.yaml](../docker-compose.yaml) file are located, so that the
  `docker compose` command can read the correct files).

- Lines 8 and 9:

  ```
  ExecStart=/path/to/docker compose up -d --build
  ExecStop=/path/to/docker compose down
  ```

  Change `/path/to/docker` to the location of the Docker executable.
  To find this location, the command `which docker` can be used.

  Also, note that the `-d` option is not used in the `docker compose` command.
  This is so that Docker does not start the container in the background, which
  would make systemd think that the process finished and would try to restart
  the container.

Where to Put the Systemd Unit File
----------------------------------

For the operating system to be aware of the service, the unit file must be
placed in the `/etc/systemd/system/` directory.

Once the unit file is in this directory, use the command:

```
sudo systemctl daemon-reload
```

to reload systemd.

Managing the Service
--------------------

- To start the service, use the following command:

  ```
  sudo systemctl start mnist_digit_predictor.service
  ```

- To enable the service, making it start whenever the computer is started
  or rebooted, use the following command:

  ```
  sudo systemctl enable mnist_digit_predictor.service
  ```

- To check the status of the service, use the following command:

  ```
  sudo systemctl status mnist_digit_predictor.service
  ```

- To view additional logs from the service, use the following command:

  ```
  journalctl -u mnist_digit_predictor.service
  ```
