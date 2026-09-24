"""m62 unit C — unit B's URL-store tests re-collected on ``s3://`` against moto.

Every unit B test that takes ``store_url`` or ``shared_url`` is imported, except the concurrent
identical puts: moto answers concurrent same-key PUTs with HTTP 500.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "url"))

from test_fsspec_store import (
    test_concurrent_record_dead_loses_nothing as test_concurrent_record_dead_loses_nothing,
)
from test_fsspec_store import test_dead_letters_keep_insertion_order as test_dead_letters_keep_insertion_order
from test_fsspec_store import test_fresh_store_is_empty as test_fresh_store_is_empty
from test_fsspec_store import test_get_missing_blob_is_none as test_get_missing_blob_is_none
from test_fsspec_store import (
    test_journal_replays_stage_deps_and_last_record_wins as test_journal_replays_stage_deps_and_last_record_wins,
)
from test_fsspec_store import test_node_writers_replay_as_a_union as test_node_writers_replay_as_a_union
from test_fsspec_store import (
    test_put_is_content_addressed_and_idempotent as test_put_is_content_addressed_and_idempotent,
)
from test_fsspec_store import (
    test_record_without_its_blob_is_not_honored as test_record_without_its_blob_is_not_honored,
)
from test_fsspec_store import (
    test_records_are_one_line_objects_with_the_local_bytes as test_records_are_one_line_objects_with_the_local_bytes,
)
from test_fsspec_store import test_unparseable_record_is_skipped as test_unparseable_record_is_skipped
from test_remote_crossprocess import (
    test_other_process_finds_nothing_left_to_do as test_other_process_finds_nothing_left_to_do,
)
from test_remote_crossprocess import (
    test_other_process_resumes_from_the_url_alone as test_other_process_resumes_from_the_url_alone,
)
from test_remote_resume import (
    test_kill_then_resume_equals_uninterrupted as test_kill_then_resume_equals_uninterrupted,
)
from test_remote_resume import (
    test_shuffle_kill_then_resume_equals_uninterrupted as test_shuffle_kill_then_resume_equals_uninterrupted,
)
from test_store_determinism import (
    test_plan_bytes_and_task_ids_do_not_depend_on_the_store as test_plan_bytes_and_task_ids_do_not_depend_on_the_store,
)
from test_url_store_seam import (
    test_root_with_glob_metacharacters_keeps_its_records as test_root_with_glob_metacharacters_keeps_its_records,
)
from test_url_store_seam import (
    test_url_get_refuses_bytes_that_do_not_hash_to_their_name as test_url_get_refuses_bytes_that_do_not_hash_to_their_name,
)
from test_url_store_seam import (
    test_url_resume_recomputes_a_corrupted_partial_and_heals_it as test_url_resume_recomputes_a_corrupted_partial_and_heals_it,
)
