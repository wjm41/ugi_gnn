import click
import rich

@click.command()
@click.option('--chunkprefix', default='chunk', help='Prefix for chunks')
@click.option('--output', default='combined.sdf.gz', help='Combined output')
def combine_chunks(chunkprefix, output):
    from glob import glob
    chunks = sorted(glob(chunkprefix))
    from openeye import oechem
    oemol = oechem.OEMol()    
    ndocked = 0
    with oechem.oemolostream(output) as ofs:
        from rich.progress import track
        for chunk in track(chunks, description=f'Assembling chunks into {output}'):
            with oechem.oemolistream(chunk) as ifs:
                while oechem.OEReadMolecule(ifs, oemol):
                    oechem.OEWriteMolecule(ofs, oemol)
                    ndocked += 1

    rich.get_console().log(f'{ndocked} docked poses written')

if __name__ == '__main__':
    combine_chunks()
