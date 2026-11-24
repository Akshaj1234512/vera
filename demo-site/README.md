# VERA demo site

The VERA pitch/demo page: concept animation, live interactive face, the Zoom interview demo, CAD, and MuJoCo clips.

Live (Berkeley sign-in, access via IAP): https://vera-demo-914670080684.us-west1.run.app

## Layout

- `static/index.html`: the whole page (HTML, CSS, JS in one file). Edit this.
- `static/posters/`, `static/assets/`: poster frames and concept images.
- `main.py`: small Flask server. Videos are not in this repo; `/m/<path>` redirects to a short-lived signed URL in the private bucket `gs://vera-demo-media-914670080684` (`media/` holds the H.264 copies the page plays).

## Add a video

```bash
gcloud storage cp my-clip.mp4 gs://vera-demo-media-914670080684/media/
```

Then reference it in `index.html` as `m/media/my-clip.mp4` (H.264 MP4 plays everywhere; convert `.mov`/HEVC/WebM first with ffmpeg).

## Deploy

GCP project `vera-demo-2026`, Cloud Run service `vera-demo`, region `us-west1`:

```bash
gcloud beta run deploy vera-demo --project vera-demo-2026 --source demo-site \
  --region us-west1 --no-allow-unauthenticated --iap --memory 512Mi --quiet
```

## Give someone access

They need both:

1. IAP access:
   ```bash
   gcloud iap web add-iam-policy-binding --project vera-demo-2026 --resource-type=cloud-run \
     --service=vera-demo --region=us-west1 --member=user:NAME@berkeley.edu \
     --role=roles/iap.httpsResourceAccessor
   ```
2. A test-user entry: Cloud Console > Google Auth Platform > Audience > Test users (project `vera-demo-2026`).
