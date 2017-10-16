import pandas as pd
import glob
import re
import os


class SpecLoader:

    defaultSpectrumColumnNames = ['WAVELENGTH',
                                  'FLUX', 'ERROR', 'CONTAM', 'ZEROTH']

    defaultCombinedGrismKey = 141 + 102

    def __init__(self,
                 baseDirPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}',
                 parDirPathPattern='{base_dir}/Par{par}',
                 withGrismSpectrumPathPattern='{par_dir}/Spectra/Par{par}_G{grism}_BEAM_{object}A.dat',
                 bothGrismSpectrumPathPattern='{par_dir}/Spectra/Par{par}_BEAM_{object}A.dat'):

        self.baseDirPathPattern = baseDirPathPattern
        self.parDirPathPattern = parDirPathPattern

        self.combinedGrismKey = SpecLoader.defaultCombinedGrismKey

        self.spectrumPathPatterns = {141: withGrismSpectrumPathPattern,
                                     102: withGrismSpectrumPathPattern,
                                     self.combinedGrismKey: bothGrismSpectrumPathPattern}
        self.pipelineVersion = None
        self.targetPar = None

        self.spectralData = {141: None,
                             102: None,
                             self.combinedGrismKey: None}

        self.spectrumFilePaths = {141: None,
                                  102: None,
                                  self.combinedGrismKey: None}

        self.spectrumDetails = {141: None,
                                102: None,
                                self.combinedGrismKey: None}

        self.spectrumFilePathsFrame = None

    def locateSingleSpectrum(self, fieldId, objectId, grism, pipelineVersion=6.2):
        baseDir = self.baseDirPathPattern.format(
            pipeline_version=pipelineVersion)
        parDirectories = glob.glob(
            self.parDirPathPattern.format(base_dir=baseDir, par='*'))
        parNumbers = [int(re.search('Par([0-9]+)', path).group(1))
                      for path in parDirectories]
        if fieldId in parNumbers:
            parDir = self.parDirPathPattern.format(
                base_dir=baseDir, par=fieldId)
            spectrumFile = self.spectrumPathPatterns[grism].format(
                par_dir=parDir, par=fieldId, grism=grism, object=objectId)
            if os.path.exists(spectrumFile):
                return spectrumFile
        return None

    def parseSpectrumHeader(self, fileName):
        with open(fileName) as catFile:
            for line in catFile.readlines():
                if line.startswith('#'):
                    return line.split()[1:]
        return None

    def loadSingleSpectrumFromPath(self, spectrumPath):
        return None if spectrumPath is None else pd.read_csv(filepath_or_buffer=spectrumPath,
                                                             delim_whitespace=True,
                                                             header=None,
                                                             names=self.parseSpectrumHeader(
                                                                 spectrumPath),
                                                             comment='#').dropna()

    def loadSingleSpectrum(self, fieldId, objectId, grism, pipelineVersion=6.2):
        spectrumPath = self.locateSingleSpectrum(
            fieldId, objectId, grism, pipelineVersion)
        return None if spectrumPath is None else pd.read_csv(filepath_or_buffer=spectrumPath,
                                                             delim_whitespace=True,
                                                             header=None,
                                                             names=self.parseSpectrumHeader(
                                                                 spectrumPath),
                                                             comment='#').dropna()

    def getSingleSpectrumSummary(self, fieldId, objectId, grism, withSpectrum=True, pipelineVersion=6.2):
        spectrum = self.loadSingleSpectrum(
            fieldId, objectId, grism, pipelineVersion)
        if spectrum is not None:
            description = spectrum.describe()
            return (description, spectrum) if withSpectrum else description

    def locateAllSpectra(self, pipelineVersion=6.2):
        baseDir = self.baseDirPathPattern.format(
            pipeline_version=pipelineVersion)
        parDirectories = glob.glob(
            self.parDirPathPattern.format(base_dir=baseDir, par='*'))
        parNumbers = [int(re.search('Par([0-9]+)', path).group(1))
                      for path in parDirectories]

        self.spectrumFilePaths = {grismKey: {parNumber: {int(re.search('BEAM_([0-9]+)', path).group(1)): path for path in
                                                         glob.glob(grismPattern.format(
                                                             par_dir=parDirectory, par=parNumber, grism=grismKey, object='*'))} for parNumber, parDirectory in zip(parNumbers, parDirectories)}
                                  if isinstance(grismKey, int) else
                                  {parNumber: {int(re.search('BEAM_([0-9]+)', path).group(1)): path for path in
                                               glob.glob(grismPattern.format(
                                                   par_dir=parDirectory, par=parNumber, object='*'))} for parNumber, parDirectory in zip(parNumbers, parDirectories)}
                                  for grismKey, grismPattern in self.spectrumPathPatterns.items()}

        spectrumPathIndexDataDict = {(int(grism), int(parNumber), int(target)): path
                                     for grism, grismData in self.spectrumFilePaths.items()
                                     for parNumber, parData in grismData.items()
                                     for target, path in parData.items()}

        self.spectrumFilePathsFrame = pd.DataFrame({'PATH': [path for path in spectrumPathIndexDataDict.values()]}, index=pd.MultiIndex.from_tuples(
            spectrumPathIndexDataDict.keys(), names=['GRISM', 'PAR', 'TARGET'])).sort_index()
