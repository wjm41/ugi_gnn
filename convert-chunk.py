# Convert each chunk independently

import click
import rich

@click.command()
@click.option('--chunkprefix', default='ugis', help='Prefix for chunks')
@click.option('--input', default='output', help='Destination directory for .bz2')
@click.option('--output', default='docked', help='Destination directory for .bz2')
@click.option('--index', default=0, help='Chunk index')
def convert_chunk(chunkprefix, input, output, index):
    chunk_filename = f'{input}/{chunkprefix}-{index:08d}.oeb'
    output_filename = f'{output}/{chunkprefix}-{index:08d}.sdf.bz2'
    
    import os
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    from openeye import oechem
    oemol = oechem.OEMol()    

    ndocked = 0
    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_SDF)
    ofs.openstring()
    with oechem.oemolistream(chunk_filename) as ifs:
        while oechem.OEReadMolecule(ifs, oemol):
            oechem.OEWriteMolecule(ofs, oemol)
            ndocked += 1

    rich.get_console().log(f'{ndocked} docked poses read')

    import bz2
    with bz2.open(output_filename, 'wt') as outfile:
        outfile.write(ofs.GetString().decode('UTF-8'))

    rich.get_console().log(f'{ndocked} docked poses written')

if __name__ == '__main__':
    convert_chunk()
