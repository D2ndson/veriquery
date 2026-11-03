"""Extract BIRD SQLite files, including dev.zip's nested dev_databases.zip."""
from pathlib import Path
import shutil
import tempfile
import zipfile


def extract_databases(archive_path, destination):
    destination = Path(destination)
    files = []
    def unpack(archive, depth=0):
        for member in archive.infolist():
            if member.filename.startswith('__MACOSX'):
                continue
            name = Path(member.filename).name
            if name.endswith('.sqlite'):
                folder = destination/Path(name).stem
                folder.mkdir(parents=True, exist_ok=True)
                target = folder/name
                if target.exists():
                    raise FileExistsError(f'Refusing to replace database {target}')
                with archive.open(member) as src, target.open('wb') as out:
                    shutil.copyfileobj(src,out)
                files.append(target)
            elif name == 'dev_databases.zip' and depth == 0:
                with tempfile.TemporaryFile() as tmp:
                    with archive.open(member) as src:
                        shutil.copyfileobj(src,tmp)
                    tmp.seek(0)
                    with zipfile.ZipFile(tmp) as nested:
                        unpack(nested,depth+1)
    with zipfile.ZipFile(archive_path) as archive:
        unpack(archive)
    if not files:
        raise ValueError('Archive contained no SQLite databases')
    return files


if __name__ == '__main__':
    import sys
    print([str(p) for p in extract_databases(sys.argv[1],sys.argv[2])])
