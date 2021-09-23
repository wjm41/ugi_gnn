#!/usr/bin/env python

"""
Process CSV file from Alpha, which is oddly misformatted
"""

import rich

def read_chunk(filename, chunk_index, chunk_size=2500):
    """
    Read SMILES from CSV file sent by Alpha Lee

    Format is weird:

    ,SMILES
    0,"0,CNC(=O)C(c1cnccc1C)N(C(=O)c1ccns1)c1ccc(C(F)(F)F)nn1"
    1,"1,CCC(C)NC(=O)C(c1cnccc1C)N(C(=O)c1ccns1)c1ccc(C(F)(F)F)nn1"
    2,"2,COCCCNC(=O)C(c1cnccc1C)N(C(=O)c1ccns1)c1ccc(C(F)(F)F)nn1"

    Parameters
    ----------
    filename : str
        The filename to read
    chunk_index : int
        Chunk index (0, 1, 2...) to read
    chunk_size : int, optional, default=100
        Chunk size to read
    """
    start_line = chunk_index * chunk_size
    end_line = (chunk_index+1) * chunk_size
    import re, linecache
    molecules = dict()
    from rich import progress
    for line_index in progress.track(range(start_line, end_line), description=f'Reading SMILES from chunk {chunk_index} {filename} using chunksize {chunk_size}'):
        line = linecache.getline(filename, line_index)
        matches = re.match('^\d+,\"(\d+),(\S+)\"$', line.strip())
        if matches is not None:
            title = matches.group(1)
            smiles = matches.group(2)                
            molecules[title] = smiles

    return molecules

def expand_stereochemistry(mol):
    """Expand stereochemistry when uncertain

    Parameters
    ----------
    mols : openeye.oechem.OEGraphMol
        Molecules to be expanded

    Returns
    -------
    expanded_mols : openeye.oechem.OEMol
        Expanded molecules
    """
    expanded_mols = list()

    from openeye import oechem, oeomega
    omegaOpts = oeomega.OEOmegaOptions()
    omega = oeomega.OEOmega(omegaOpts)
    maxcenters = 12
    forceFlip = False
    enumNitrogen = True
    warts = True # add suffix for stereoisomers
    mols = [mol]
    for mol in mols:
        compound_title = mol.GetTitle()
        compound_smiles = oechem.OEMolToSmiles(mol)

        enantiomers = list()
        for enantiomer in oeomega.OEFlipper(mol, maxcenters, forceFlip, enumNitrogen, warts):
            enantiomer = oechem.OEMol(enantiomer)
            enantiomer_smiles =  oechem.OEMolToSmiles(enantiomer)
            oechem.OESetSDData(enantiomer, 'compound', compound_title)
            oechem.OESetSDData(enantiomer, 'compound_smiles', compound_smiles)
            oechem.OESetSDData(enantiomer, 'enantiomer_smiles', enantiomer_smiles)
            enantiomers.append(enantiomer)

        expanded_mols += enantiomers

    return expanded_mols

class BumpCheck:
    def __init__(self, prot_mol, cutoff=2.0):
        from openeye import oechem
        self.near_nbr = oechem.OENearestNbrs(prot_mol, cutoff)
        self.cutoff = cutoff

    def count(self, lig_mol):
        import numpy as np
        bump_count = 0
        for nb in self.near_nbr.GetNbrs(lig_mol):
            if (not nb.GetBgn().IsHydrogen()) and (not nb.GetEnd().IsHydrogen()):
                bump_count += np.exp(-0.5 * (nb.GetDist() / self.cutoff)**2)
        return bump_count

def mmff_energy(mol):
    """
    Compute MMFF energy

    """
    from openeye import oechem, oeff
    mol = oechem.OEGraphMol(mol)
    mmff = oeff.OEMMFF()
    if not mmff.PrepMol(mol) or not mmff.Setup(mol):
        oechem.OEThrow.Warning("Unable to process molecule: title = '%s'" % mol.GetTitle())
        return None

    vecCoords = oechem.OEDoubleArray(3*mol.GetMaxAtomIdx())
    mol.GetCoords(vecCoords)
    energy = mmff(vecCoords)
    return energy

def generate_restricted_conformers(receptor, refmol, mol, core_smarts=None):
    """
    Generate and select a conformer of the specified molecule using the reference molecule

    Parameters
    ----------
    receptor : openeye.oechem.OEDesignUnit
        Design unit containing receptor (already prepped for docking) for identifying optimal pose
    refmol : openeye.oechem.OEGraphMol
        Reference molecule which shares some part in common with the proposed molecule
    mol : openeye.oechem.OEGraphMol
        Molecule whose conformers are to be enumerated
    core_smarts : str, optional, default=None
        If core_smarts is specified, substructure will be extracted using SMARTS.

    Returns
    -------
    docked_molecules : list of OEMol
        One or more docked molecules
    """
    from openeye import oechem, oeomega
    console = rich.get_console()

    console.log(f'mol: {oechem.OEMolToSmiles(mol)} | core_smarts: {core_smarts}')

    # Be quiet
    from openeye import oechem
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)
    #oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)

    # Get core fragment
    core_fragment = None
    if core_smarts:
        # Truncate refmol to SMARTS if specified
        #print(f'Trunctating using SMARTS {refmol_smarts}')
        ss = oechem.OESubSearch(core_smarts)
        oechem.OEPrepareSearch(refmol, ss)
        for match in ss.Match(refmol):
            core_fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(core_fragment, match)
            console.log(f'Truncated refmol to generate core_fragment: {oechem.OEMolToSmiles(core_fragment)}')
            break
        #print(f'refmol has {refmol.NumAtoms()} atoms')
    else:
        core_fragment = GetCoreFragment(refmol, [mol])
        oechem.OESuppressHydrogens(core_fragment)
        #print(f'  Core fragment has {core_fragment.NumAtoms()} heavy atoms')
        MIN_CORE_ATOMS = 6
        if core_fragment.NumAtoms() < MIN_CORE_ATOMS:
            return None

    if core_fragment is None:
        return None

    # Create an Omega instance
    #omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)

    # Set the fixed reference molecule
    omegaFixOpts = oeomega.OEConfFixOptions()
    omegaFixOpts.SetFixMaxMatch(10) # allow multiple MCSS matches
    omegaFixOpts.SetFixDeleteH(True) # only use heavy atoms
    omegaFixOpts.SetFixMol(core_fragment)
    #omegaFixOpts.SetFixSmarts(core_smarts) # DEBUG
    omegaFixOpts.SetFixRMS(0.5)

    # This causes a warning:
    #Warning: OESubSearch::Match() is unable to match unset hybridization in the target (EN300-221518_3_1) for patterns with set hybridization, call OEPrepareSearch on the target first
    #atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_Hybridization

    atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_AtomicNumber
    bondexpr = oechem.OEExprOpts_BondOrder | oechem.OEExprOpts_Aromaticity
    omegaFixOpts.SetAtomExpr(atomexpr)
    omegaFixOpts.SetBondExpr(bondexpr)
    omegaOpts.SetConfFixOptions(omegaFixOpts)

    molBuilderOpts = oeomega.OEMolBuilderOptions()
    molBuilderOpts.SetStrictAtomTypes(False) # don't give up if MMFF types are not found
    omegaOpts.SetMolBuilderOptions(molBuilderOpts)

    omegaOpts.SetWarts(False) # expand molecule title
    omegaOpts.SetStrictStereo(True) # set strict stereochemistry
    omegaOpts.SetIncludeInput(False) # don't include input
    omegaOpts.SetMaxConfs(1000) # generate lots of conformers
    omegaOpts.SetEnergyWindow(20.0) # allow high energies
    omega = oeomega.OEOmega(omegaOpts)

    # Expand to multi-conformer molecule
    mol = oechem.OEMol(mol) # multi-conformer molecule

    ret_code = omega.Build(mol)
    if (mol.GetDimension() != 3) or (ret_code != oeomega.OEOmegaReturnCode_Success):
        msg = f'\nOmega failure for {mol.GetTitle()} : SMILES {oechem.OEMolToSmiles(mol)} : core_smarts {core_smarts} : {oeomega.OEGetOmegaError(ret_code)}\n'
        console.log(msg)
        return None
        # Return the molecule with an error code
        #oechem.OESetSDData(mol, 'error', '{oeomega.OEGetOmegaError(ret_code)}')
        #return mol

    # Extract poses
    class Pose(object):
        def __init__(self, conformer):
            self.conformer = conformer
            self.clash_score = None
            self.docking_score = None
            self.overlap_score = None

    poses = [ Pose(conf) for conf in mol.GetConfs() ]

    # Score clashes
    bump_check = BumpCheck(receptor)
    for pose in poses:
        pose.clash_score = bump_check.count(pose.conformer)

    # Score docking poses
    from openeye import oedocking
    score = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
    score.Initialize(receptor)
    for pose in poses:
        pose.docking_score = score.ScoreLigand(pose.conformer)

    # Compute overlap scores
    from openeye import oeshape
    overlap_prep = oeshape.OEOverlapPrep()
    overlap_prep.Prep(refmol)
    shapeFunc = oeshape.OEExactShapeFunc()
    shapeFunc.SetupRef(refmol)
    oeshape_result = oeshape.OEOverlapResults()
    for pose in poses:
        tmpmol = oechem.OEGraphMol(pose.conformer)
        overlap_prep.Prep(tmpmol)
        shapeFunc.Overlap(tmpmol, oeshape_result)
        pose.overlap_score = oeshape_result.GetRefTversky()

    # Filter poses based on top 10% of overlap
    poses = sorted(poses, key= lambda pose : pose.overlap_score)
    poses = poses[int(0.9*len(poses)):]

    # Select the best docking score
    import numpy as np
    poses = sorted(poses, key=lambda pose : pose.docking_score)
    pose = poses[0]
    mol.SetActive(pose.conformer)
    oechem.OESetSDData(mol, 'clash_score', str(pose.clash_score))
    oechem.OESetSDData(mol, 'docking_score', str(pose.docking_score))
    oechem.OESetSDData(mol, 'overlap_score', str(pose.overlap_score))

    # Convert to single-conformer molecule
    mol = oechem.OEGraphMol(mol)

    # Compute MMFF energy
    energy = mmff_energy(mol)
    oechem.OESetSDData(mol, 'MMFF_internal_energy', str(energy))

    # Store SMILES
    docked_smiles = oechem.OEMolToSmiles(mol)
    oechem.OESetSDData(mol, 'docked_smiles', docked_smiles)

    return mol

import click
@click.command()
@click.option('--csvfile', 'csv_filename', default='total_ugi_library.csv', help='CSV file containing SMILES')
@click.option('--fixsmarts', default='C(=O)NCC(=O)N', help='SMARTS string to fix when enumerating conformers')
@click.option('--receptor', 'receptor_filename', default='receptors/dimer/Mpro-N0050-receptor.oeb.gz', help='OE receptor design unit (.oeb.gz)')
@click.option('--refmol', 'refmol_filename', default='receptors/dimer/Mpro-N0050-ligand.mol2', help='Reference ligand (.mol2)')
@click.option('--chunk', 'chunk_index', default=0, help='Chunk index (zero-indexed)')
@click.option('--chunksize', 'chunk_size', default=2500, help='Chunk size (default: 2500)')
@click.option('--output', default='output', help='Output prefix')
def dock_chunk(csv_filename, fixsmarts, receptor_filename, refmol_filename, chunk_index, chunk_size, output):
    console = rich.get_console()

    # Read SMILES
    console.rule("[bold red]Reading SMILES")
    molecules = read_chunk(csv_filename, chunk_index=chunk_index, chunk_size=chunk_size)
    console.log(f'{len(molecules)} molecules read.')

    # Dock compounds in chunk
    console.rule("[bold red]Docking")
    from openeye import oechem    

    # Read the receptor
    from openeye import oedocking
    receptor = oechem.OEGraphMol()
    oedocking.OEReadReceptorFile(receptor, receptor_filename)
    console.log(f'Receptor has {receptor.NumAtoms()} atoms.')
    # Read the reference molecule
    refmol = oechem.OEMol()
    with oechem.oemolistream(refmol_filename) as ifs:
        oechem.OEReadMolecule(ifs, refmol)
    console.log(f'Reference ligand has {refmol.NumAtoms()} atoms.')

    # Dock all molecules in chunk
    ndocked = 0
    nconsidered = 0
    with oechem.oemolostream(f'{output}-{chunk_index:08d}.oeb') as ofs:

        from rich.progress import Progress, BarColumn, TimeRemainingColumn
        with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task('Docking molecules', total=len(molecules), ndocked=0)
            for title, smiles in molecules.items():
                # Create molecule
                oemol = oechem.OEMol()
                oechem.OESmilesToMol(oemol, smiles)
                oemol.SetTitle(title)
                # Expand protonation states
                from openeye import oequacpac
                for protonation_state in oequacpac.OEGetReasonableProtomers(oemol):
                    # Expand stereochemistry
                    stereoisomers = expand_stereochemistry(protonation_state)
                    for stereoisomer in stereoisomers:
                        # Dock molecule
                        docked_molecule = generate_restricted_conformers(receptor, refmol, stereoisomer, core_smarts=fixsmarts)
                        # Write it if a docked molecule was generated
                        if docked_molecule is not None:
                            oechem.OEWriteMolecule(ofs, docked_molecule)
                            ndocked += 1
                            progress.log(f'{ndocked} docked poses generated')
                # Update number of molecules considered
                nconsidered += 1
                progress.log(f'{nconsidered} / {chunk_size} molecules considered')
                # Update progress bar
                progress.advance(task)

        
            
    console.log('Completed.')


if __name__ == '__main__':
    dock_chunk()
