import datalad.api
import pathlib
import tempfile
from filelock import FileLock


def expr_to_opts(expr):
    opts = []
    expr = expr.replace('(', ' ( ').replace(')', ' ) ')
    for sub_expr in expr.split(' '):
        if len(sub_expr):
            if sub_expr in '()':
                opts.append(f"-{sub_expr}")
            else:
                opts.append(f"--{sub_expr}")
    return opts


def install_ria_lock(ria_url, dest_path, **kwargs):
    remote_path = pathlib.Path(ria_url.replace('ria+file://', '').replace('#~', '/alias/').split('@')[0])
    lock_path = remote_path / '.datalad_lock'
    file_lock = FileLock(lock_path)

    with file_lock:
        ds = datalad.api.install(path=dest_path, source=ria_url, **kwargs)
    return ds

def ria_fetch(ria_url, remote):
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds = install_ria_lock(ria_url, tmpdirname, reckless='ephemeral')
        try:
            ds.repo.enable_remote(remote)
        except AccessFailedError as e:
            #TODO: handle local to ssh url fix
            pass
        
        ria_storage_remote = ds.config.get('remote.origin.datalad-publish-depends')
        # get all wanted data, as cloned as ephemeral, will get data in the RIA store
        wanted = ds.repo.get_preferred_content('wanted', remote=ria_storage_remote) or ''
        ds.repo.call_annex(['get', '--all', '--from', remote] + expr_to_opts(wanted))
        # then fsck on the ria remote
        ds.repo.fsck(remote=ria_storage_remote, fast=True)
        # push the git-annex branch to the RIA for updated file locations
        ds.push(to='origin', data='nothing')
    
    
def ria_push(ria_url, remote):
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds = install_ria_lock(ria_url, tmpdirname, reckless='ephemeral')
        # get all wanted data, as cloned as ephemeral, will get data in the RIA store
        ds.push(to='remote')
