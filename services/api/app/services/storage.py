from __future__ import annotations

from pathlib import Path

import boto3
from botocore.client import Config

from app.core.config import get_settings


class ReportStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.local_root = Path(self.settings.report_local_dir).resolve()
        self.local_root.mkdir(parents=True, exist_ok=True)

    @property
    def s3_enabled(self) -> bool:
        return all(
            [
                self.settings.r2_endpoint,
                self.settings.r2_bucket,
                self.settings.r2_access_key,
                self.settings.r2_secret_key,
            ]
        )

    def _client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.settings.r2_endpoint,
            region_name=self.settings.r2_region,
            aws_access_key_id=self.settings.r2_access_key,
            aws_secret_access_key=self.settings.r2_secret_key,
            config=Config(signature_version="s3v4"),
        )

    def save_local(self, report_id: str, filename: str, data_path: Path) -> str:
        target = self.local_root / report_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data_path.read_bytes())
        return f"{self.settings.public_base_url}/v1/reports/files/{report_id}/{filename}"

    def upload(self, report_id: str, filename: str, data_path: Path) -> str:
        if not self.s3_enabled:
            return self.save_local(report_id, filename, data_path)

        key = f"reports/{report_id}/{filename}"
        client = self._client()
        client.upload_file(str(data_path), self.settings.r2_bucket, key)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.r2_bucket, "Key": key},
            ExpiresIn=3600,
        )


report_storage = ReportStorage()

