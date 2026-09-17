// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

module.exports = async ({ github, context, core }) => {
  if (!/^refs\/tags\/\d+\.\d+\.\d+$/.test(context.ref) || !/^[a-f0-9]{40}$/.test(context.sha)) {
    throw new Error('Main staging verification requires a stable tag and full source commit.');
  }
  const deadline = Date.now() + 45 * 60 * 1000;
  while (true) {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...context.repo, workflow_id: 'snapshot.yml', branch: 'main',
      event: 'workflow_dispatch', head_sha: context.sha, per_page: 100,
    });
    const matching = runs.filter(run =>
      run.path === '.github/workflows/snapshot.yml' && run.event === 'workflow_dispatch' &&
      run.head_repository?.full_name === `${context.repo.owner}/${context.repo.repo}` &&
      run.head_branch === 'main' && run.head_sha === context.sha);

    for (const run of matching.filter(candidate => candidate.status === 'completed')) {
      const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
        ...context.repo, run_id: run.id, filter: 'all', per_page: 100,
      });
      // Failed-job retries can reuse a successful staging check from an earlier attempt.
      const verified = jobs.filter(job => job.name === 'SDK Staging Verified' &&
        job.run_attempt <= run.run_attempt)
        .sort((left, right) => right.run_attempt - left.run_attempt)[0];
      if (verified?.status === 'completed' && verified.conclusion === 'success' &&
          verified.head_sha === context.sha) {
        core.setOutput('run_id', String(run.id));
        core.info(`Main staging verified ${context.sha}: ${verified.html_url}`);
        await core.summary.addLink('Successful main staging verification', verified.html_url).write();
        return;
      }
    }

    if (matching.length && matching.every(run => run.status === 'completed')) {
      throw new Error(`No successful SDK staging verification on main for ${context.sha}. ` +
        'Complete the main Snapshot staging checks, then rerun the failed release jobs.');
    }
    if (Date.now() >= deadline) {
      throw new Error(`Timed out waiting for main staging verification of ${context.sha}. ` +
        'Complete the main Snapshot staging checks, then rerun the failed release jobs.');
    }
    core.info(`Waiting for the main Snapshot staging verification of ${context.sha}.`);
    await new Promise(resolve => setTimeout(resolve, 30000));
  }
};
