"""Serve the VERA demo page. Access control: Cloud Run IAP (Google sign-in)."""
import os
from flask import Flask, request, send_from_directory, redirect, abort
import datetime, google.auth
from google.auth.transport import requests as gareq
from google.cloud import storage
app = Flask(__name__, static_folder=None)
ROOT = os.path.join(os.path.dirname(__file__), 'static')

@app.after_request
def headers(r):
    r.headers['X-Robots-Tag'] = 'noindex, nofollow'
    # pages always revalidate so a redeploy shows up on refresh; media can be cached
    html_page = request.path == '/' or request.path.endswith('.html')
    r.headers['Cache-Control'] = 'private, no-cache' if html_page else 'private, max-age=300'
    return r

BUCKET = os.environ.get('MEDIA_BUCKET', 'vera-demo-media-914670080684')
_gcs = None


@app.route('/m/<path:p>')
def media(p):
    """Large media lives in a private GCS bucket; hand out a short-lived signed URL."""
    global _gcs
    if '..' in p:
        abort(400)
    creds, _ = google.auth.default()
    creds.refresh(gareq.Request())
    if _gcs is None:
        _gcs = storage.Client(credentials=creds)
    blob = _gcs.bucket(BUCKET).blob(p)
    url = blob.generate_signed_url(version='v4', expiration=datetime.timedelta(hours=2), method='GET',
                                   service_account_email=creds.service_account_email, access_token=creds.token)
    r = redirect(url, 302)
    r.headers['Cache-Control'] = 'private, max-age=600'
    return r


@app.route('/')
def index():
    return send_from_directory(ROOT, 'index.html')

@app.route('/<path:p>')
def files(p):
    return send_from_directory(ROOT, p, conditional=True)
