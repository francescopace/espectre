// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

module.exports = async ({ github, context, core, publication }) => {
  const inputs = context.payload.inputs || {};
  const runId = Number(inputs.source_run_id);
  const attempt = Number(inputs.source_run_attempt);
  const sha = inputs.source_sha;
  if (context.eventName !== 'workflow_dispatch' ||
      !Number.isSafeInteger(runId) || runId <= 0 ||
      !Number.isSafeInteger(attempt) || attempt <= 0 || !/^[a-f0-9]{40}$/.test(sha || '')) {
    throw new Error('Publication requires a CI run ID, attempt, and full source commit.');
  }
  const snapshot = publication === 'snapshot';
  if (snapshot ? !['refs/heads/main', 'refs/heads/develop'].includes(context.ref)
    : publication !== 'release' || !/^refs\/tags\/\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(context.ref)) {
    throw new Error('The dispatch reference does not match the publication channel.');
  }

  // CI dispatches this workflow without waiting; let its final job finish first.
  const deadline = Date.now() + 4 * 60 * 1000;
  let source;
  while (true) {
    ({ data: source } = await github.rest.actions.getWorkflowRun({
      ...context.repo, run_id: runId,
    }));
    if (source.path !== '.github/workflows/ci.yml' || source.event !== 'push' ||
        source.head_repository?.full_name !== `${context.repo.owner}/${context.repo.repo}` ||
        source.head_sha !== sha || source.run_attempt !== attempt ||
        context.ref !== `refs/${snapshot ? 'heads' : 'tags'}/${source.head_branch}`) {
      throw new Error('The source must be the matching CI push run, commit, ref, and attempt.');
    }
    if (source.status === 'completed') break;
    if (Date.now() >= deadline) throw new Error('Timed out waiting for the source CI run to finish.');
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  if (source.conclusion !== 'success') {
    throw new Error(`The source CI run cannot publish: ${source.conclusion}.`);
  }
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...context.repo, run_id: runId, filter: 'all', per_page: 100,
  });
  // Failed-job retries can reuse successful prerequisite jobs from an earlier attempt.
  const latest = name => jobs.filter(job => job.name === name && job.run_attempt <= attempt)
    .sort((left, right) => right.run_attempt - left.run_attempt)[0];
  if (latest('Dispatch Publication')?.conclusion !== 'success' ||
      latest('Build SDK Package')?.conclusion !== 'success') {
    throw new Error('The source CI run must pass its checks and authorize publication.');
  }

  let current = true;
  if (snapshot) {
    const { data: branch } = await github.rest.repos.getBranch({
      ...context.repo, branch: source.head_branch,
    });
    current = branch.commit.sha === sha;
  }
  if (current && context.sha !== sha) {
    throw new Error('The dispatched workflow must use the validated source revision.');
  }
  core.setOutput('current', String(current));
  core.setOutput('run_id', String(runId));
  core.setOutput('head_sha', sha);
  core.setOutput('head_branch', source.head_branch);
  core.setOutput('created_at', source.created_at);
  core.setOutput('conclusion', source.conclusion);
  const channel = snapshot ? (source.head_branch === 'develop' ? 'develop' : 'preview') : 'release';
  const stable = !snapshot && !source.head_branch.includes('-');
  core.setOutput('metadata', JSON.stringify({
    run_id: String(runId), head_sha: sha, head_branch: source.head_branch,
    created_at: source.created_at, publication, channel, stable,
    release_tag: snapshot ? (channel === 'develop' ? 'snapshot-dev' : 'snapshot') : source.head_branch,
    registry_url: stable ? 'https://components.espressif.com' : 'https://components-staging.espressif.com',
    environment: stable ? 'sdk-registry-production' : 'sdk-registry-staging',
    publish_website: !snapshot || source.head_branch === 'main',
  }));
  core.info(current ? `Source CI: ${source.html_url} (${source.conclusion}).` : 'Skipping a superseded snapshot.');
  await core.summary.addLink('Source CI run', source.html_url).write();
};
