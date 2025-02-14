Downloading shards:   0%|          | 0/8 [00:00<?, ?it/s]Downloading shards:   0%|          | 0/8 [2:14:14<?, ?it/s]
Traceback (most recent call last):
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/utils/_http.py", line 406, in hf_raise_for_status
    response.raise_for_status()
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://cdn-lfs-us-1.hf.co/repos/67/34/673427938994a8d86ada158434b6063cd39c1f5897ee716799bd60defdf9bcf0/52addd84f7e83b6f5b54edfbcd11a9dc71ced13e63967e0e551fcdd071a09cac?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27model-00001-of-000008.safetensors%3B+filename%3D%22model-00001-of-000008.safetensors%22%3B&Expires=1739475617&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTQ3NTYxN319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzY3LzM0LzY3MzQyNzkzODk5NGE4ZDg2YWRhMTU4NDM0YjYwNjNjZDM5YzFmNTg5N2VlNzE2Nzk5YmQ2MGRlZmRmOWJjZjAvNTJhZGRkODRmN2U4M2I2ZjViNTRlZGZiY2QxMWE5ZGM3MWNlZDEzZTYzOTY3ZTBlNTUxZmNkZDA3MWEwOWNhYz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=a2RII47ACEG76q7u3KKS1GJk1AINhuaKTpTWtVxJgjwUA-i6kPaYj5gdv20ESJ8heuCh1y~rpsF7IfEC~58rSq0089btVXLPJpkUxF3v9SxFKPtG2659P6Snc8EjRbVHORQwxCjrd0cG3kPRAUBq3c0fDihFFdSjqyYsj9qFVeusIxb~B8uxT9Dy3685Geml6aJOqbSYH00whdRnsVixjN7Ft6ye5-7iVq1Ok7Vu4yRozhuJuMEaoYnrsUN2mazDJwnpLTbVXu6JdO4NzrKK5pXf1GXd61EogWlwXm6~6S1XUXqkhZ1RXMcN1QBq7YkYftQPZSX-2dTywOhTefmzew__&Key-Pair-Id=K24J24Z295AEI9

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/transformers/utils/hub.py", line 403, in cached_file
    resolved_file = hf_hub_download(
                    ^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 860, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1009, in _hf_hub_download_to_cache_dir
    _download_to_tmp_and_move(
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1543, in _download_to_tmp_and_move
    http_get(
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 369, in http_get
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 302, in _request_wrapper
    hf_raise_for_status(response)
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/huggingface_hub/utils/_http.py", line 468, in hf_raise_for_status
    raise _format(HfHubHTTPError, message, response) from e
huggingface_hub.errors.HfHubHTTPError: 403 Forbidden: None.
Cannot access content at: https://cdn-lfs-us-1.hf.co/repos/67/34/673427938994a8d86ada158434b6063cd39c1f5897ee716799bd60defdf9bcf0/52addd84f7e83b6f5b54edfbcd11a9dc71ced13e63967e0e551fcdd071a09cac?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27model-00001-of-000008.safetensors%3B+filename%3D%22model-00001-of-000008.safetensors%22%3B&Expires=1739475617&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTQ3NTYxN319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzY3LzM0LzY3MzQyNzkzODk5NGE4ZDg2YWRhMTU4NDM0YjYwNjNjZDM5YzFmNTg5N2VlNzE2Nzk5YmQ2MGRlZmRmOWJjZjAvNTJhZGRkODRmN2U4M2I2ZjViNTRlZGZiY2QxMWE5ZGM3MWNlZDEzZTYzOTY3ZTBlNTUxZmNkZDA3MWEwOWNhYz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=a2RII47ACEG76q7u3KKS1GJk1AINhuaKTpTWtVxJgjwUA-i6kPaYj5gdv20ESJ8heuCh1y~rpsF7IfEC~58rSq0089btVXLPJpkUxF3v9SxFKPtG2659P6Snc8EjRbVHORQwxCjrd0cG3kPRAUBq3c0fDihFFdSjqyYsj9qFVeusIxb~B8uxT9Dy3685Geml6aJOqbSYH00whdRnsVixjN7Ft6ye5-7iVq1Ok7Vu4yRozhuJuMEaoYnrsUN2mazDJwnpLTbVXu6JdO4NzrKK5pXf1GXd61EogWlwXm6~6S1XUXqkhZ1RXMcN1QBq7YkYftQPZSX-2dTywOhTefmzew__&Key-Pair-Id=K24J24Z295AEI9.
Make sure your token has the correct permissions.
<?xml version="1.0" encoding="UTF-8"?><Error><Code>AccessDenied</Code><Message>Access denied</Message></Error>

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/users/aczd097/git/vLLM/DeepSeek-R1-Distill-Qwen-32B.py", line 23, in <module>
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/transformers/models/auto/auto_factory.py", line 564, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/transformers/modeling_utils.py", line 3944, in from_pretrained
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/transformers/utils/hub.py", line 1098, in get_checkpoint_shard_files
    cached_filename = cached_file(
                      ^^^^^^^^^^^^
  File "/users/aczd097/.pyenv/versions/trans-env/lib/python3.11/site-packages/transformers/utils/hub.py", line 467, in cached_file
    raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}")
OSError: There was a specific connection error when trying to load deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:
403 Forbidden: None.
Cannot access content at: https://cdn-lfs-us-1.hf.co/repos/67/34/673427938994a8d86ada158434b6063cd39c1f5897ee716799bd60defdf9bcf0/52addd84f7e83b6f5b54edfbcd11a9dc71ced13e63967e0e551fcdd071a09cac?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27model-00001-of-000008.safetensors%3B+filename%3D%22model-00001-of-000008.safetensors%22%3B&Expires=1739475617&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTQ3NTYxN319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzY3LzM0LzY3MzQyNzkzODk5NGE4ZDg2YWRhMTU4NDM0YjYwNjNjZDM5YzFmNTg5N2VlNzE2Nzk5YmQ2MGRlZmRmOWJjZjAvNTJhZGRkODRmN2U4M2I2ZjViNTRlZGZiY2QxMWE5ZGM3MWNlZDEzZTYzOTY3ZTBlNTUxZmNkZDA3MWEwOWNhYz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=a2RII47ACEG76q7u3KKS1GJk1AINhuaKTpTWtVxJgjwUA-i6kPaYj5gdv20ESJ8heuCh1y~rpsF7IfEC~58rSq0089btVXLPJpkUxF3v9SxFKPtG2659P6Snc8EjRbVHORQwxCjrd0cG3kPRAUBq3c0fDihFFdSjqyYsj9qFVeusIxb~B8uxT9Dy3685Geml6aJOqbSYH00whdRnsVixjN7Ft6ye5-7iVq1Ok7Vu4yRozhuJuMEaoYnrsUN2mazDJwnpLTbVXu6JdO4NzrKK5pXf1GXd61EogWlwXm6~6S1XUXqkhZ1RXMcN1QBq7YkYftQPZSX-2dTywOhTefmzew__&Key-Pair-Id=K24J24Z295AEI9.
Make sure your token has the correct permissions.
<?xml version="1.0" encoding="UTF-8"?><Error><Code>AccessDenied</Code><Message>Access denied</Message></Error>
