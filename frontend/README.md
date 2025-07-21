MNIST Digit Prediction Frontend
===============================

The frontend for the MNIST digit prediction system is a basic HTML, CSS, JavaScript project bundled with Vite.

Installation
------------

Make sure you are in the frontend directory:

```bash
cd frontend
```

Install Npm packages:

```bash
npm install
```

To run the development server you can use an Npm script defined in [package.json](./package.json):

```bash
npm run dev
```

This should run the live server. Any changes you make to the frontend files will get updated without you having to restart the server.

If you want to use custom modes, you can use the Vite command directly

```bash
npx vite --mode development
```

This imports the environment variables from [.env.development](./.env.development).

Building and Deploying
----------------------

To build the frontend, use the Vite command directly so that you can specify a custom mode:

```bash
npx vite build --mode production
```

Make sure you have the file `.env.production` with all the necessary environment variables.

At this point there should be a folder called `dist` with the output files. This is what will be deployed. I like using GCP cloud storage buckets for this kind of thing. Here's how you can upload to a GCP bucket called `my-bucket-name` (assuming you are still in the `frontend` directory):

```bash
gsutil -m rsync -R dist/ gs://my-bucket-name
```
