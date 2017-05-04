import pandas as pd
import numpy as np
import astropy.units as astrounits
import os
import itertools
import matplotlib.pyplot as mplplot
import glob
import re


class SpecLoader:

    def __init__(self,
                 baseDirPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}',
                 parDirPathPattern='{base_dir}/Par{par}',
                 withGrismSpectrumPathPattern='{par_dir}/Spectra/Par{par}_G{grism}_BEAM_{object}A.dat',
                 bothGrismSpectrumPathPattern='{par_dir}/Spectra/Par{par}_BEAM_{object}A.dat'):

        self.baseDirPathPattern = baseDirPathPattern
        self.parDirPathPattern = parDirPathPattern

        self.spectrumPathPatterns = {141: withGrismSpectrumPathPattern,
                                     102: withGrismSpectrumPathPattern,
                                     'COMBINED': bothGrismSpectrumPathPattern}
        self.pipelineVersion = None
        self.targetPar = None

        self.spectralData = {141: None,
                             102: None,
                             'COMBINED': None}

        self.spectrumFilePaths = {141: None,
                                  102: None,
                                  'COMBINED': None}

        self.spectrumDetails = {141: None,
                                102: None,
                                'COMBINED': None}

        self.spectrumFilePathsFrame = None

    def locateAllSpectra(self, pipelineVersion=6.2):
        baseDir = self.baseDirPathPattern.format(pipeline_version=pipelineVersion)
        parDirectories = glob.glob(self.parDirPathPattern.format(base_dir=baseDir, par='*'))
        parNumbers = [int(re.search('Par([0-9]+)', path).group(1)) for path in parDirectories]

        self.spectrumFilePaths = {grismKey: {parNumber: {int(re.search('BEAM_([0-9]+)', path).group(1)): path for path in
                                                         glob.glob(grismPattern.format(
                                                             par_dir=parDirectory, par=parNumber, grism=grismKey, object='*'))} for parNumber, parDirectory in zip(parNumbers, parDirectories)}
                                  if isinstance(grismKey, int) else
                                  {parNumber: {int(re.search('BEAM_([0-9]+)', path).group(1)): path for path in
                                               glob.glob(grismPattern.format(
                                                   par_dir=parDirectory, par=parNumber, object='*'))} for parNumber, parDirectory in zip(parNumbers, parDirectories)}
                                  for grismKey, grismPattern in self.spectrumPathPatterns.items()}

        spectrumPathIndexDataDict = {(grism, int(parNumber), target): path
                                     for grism, grismData in self.spectrumFilePaths.items()
                                     for parNumber, parData in grismData.items()
                                     for target, path in parData.items()}

        self.spectrumFilePathsFrame = pd.DataFrame({'PATH': [ path for path in spectrumPathIndexDataDict.values() ]}, index=pd.MultiIndex.from_tuples(
            spectrumPathIndexDataDict.keys(), names=['GRISM', 'PAR', 'TARGET'])).sort_index()

# TODO: Refactor to use SpecLoader


class SpecPlotter:

    spectrumLabels = {
        141: 'G141',
        102: 'G102',
        'COMBINED': 'Combined',
        'MODEL': 'Model'}

    spectrumColours = {
        141: 'red',
        102: 'blue',
        'COMBINED': 'gray',
        'MODEL': 'green'}

    spectrumRanges = {
        141: (1100, 1700) * astrounits.nanometer,  # (1075, 1700)*astrounits.nanometer,
        102: (800, 1150) * astrounits.nanometer,  # (800, 1150)*astrounits.nanometer
        'COMBINED': (800, 1700) * astrounits.nanometer,
        'MODEL' : (900, 1700)*astrounits.nanometer}

    filterPivotWavelengthsForGrism = {102: (110, 11534.46 * astrounits.angstrom),
                                      141: (160, 15370.33 * astrounits.angstrom)}

    filterWavelengthRangesForGrism = {102: (110, (8832, 14121) * astrounits.angstrom),
                                      141: (160, (13854, 16999) * astrounits.angstrom)}

    plottedWavelengthRange = (8500, 16500) * astrounits.angstrom

    def __init__(self,
                 withGrismSpectrumPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/Spectra/Par{par}_G{grism}_BEAM_{object}A.dat',
                 bothGrismSpectrumPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/Spectra/Par{par}_BEAM_{object}A.dat',
                 modelSpectrumPathPattern='{model_name}'):
        self.spectrumPathPatterns = {141: withGrismSpectrumPathPattern,
                                     102: withGrismSpectrumPathPattern,
                                     'COMBINED': bothGrismSpectrumPathPattern,
                                     'MODEL': modelSpectrumPathPattern}
        self.pipelineVersion = None
        self.targetPar = None

        self.spectralData = {141: None,
                             102: None,
                             'COMBINED': None,
                             'MODEL': None}

    def loadSpectralData(self, targetObject, targetPar, bestFitModelName, bestFitModelNorm, pipelineVersion=6.2):
        # lazy assignment of targetPar and pipelineVersion
        self.targetObject = targetObject
        self.targetPar = targetPar
        self.bestFitModelName = bestFitModelName
        self.bestFitModelNorm = bestFitModelNorm
        self.pipelineVersion = pipelineVersion
        # load spectral data for individual grisms
        for grism in self.spectralData.keys():
            testSpectrumPath = self.spectrumPathPatterns[grism].format(
                pipeline_version=self.pipelineVersion,
                par=self.targetPar,
                grism=grism,
                object=targetObject,
                model_name=self.bestFitModelName)

            if os.path.isfile(testSpectrumPath):
                if 'MODEL' in grism:
                    self.spectralData[grism] = pd.read_csv(
                        testSpectrumPath,
                        header=4,
                        delim_whitespace=True,
                        names=['WAVELENGTH', 'FLUX']).dropna()
                    self.spectralData[grism].FLUX /= self.bestFitModelNorm
                    self.spectralData[grism]['ERROR'] = np.zeros_like(self.spectralData[grism].FLUX)
                else :
                    self.spectralData[grism] = pd.read_csv(
                        testSpectrumPath,
                        skipinitialspace=True,
                        header=None,
                        comment='#',
                        delim_whitespace=True,
                        names=['WAVELENGTH', 'FLUX', 'ERROR', 'CONTAM', 'ZEROTH']).dropna()

    def computePlottingLimits(self):
        minFluxes = []
        maxFluxes = []

        for grism, data in self.spectralData.items():
            if data is not None:
                grismRange = SpecPlotter.spectrumRanges[grism].to(astrounits.angstrom).value
                fullRange = SpecPlotter.plottedWavelengthRange.to(astrounits.angstrom).value
                # print (grismRange)
                validWavelengthIndices = np.where((data.WAVELENGTH < grismRange[1]) &
                                                  (data.WAVELENGTH > grismRange[0]) &
                                                  (data.WAVELENGTH > fullRange[0]) &
                                                  (data.WAVELENGTH < fullRange[1]))[0]
                # print (list(zip(validWavelengthIndices,
                # data.FLUX[validWavelengthIndices],
                # data.WAVELENGTH[validWavelengthIndices])))
                maxFluxes.append(
                    np.amax(data.FLUX[validWavelengthIndices]) + np.amax(data.ERROR[validWavelengthIndices]))
                minFluxes.append(
                    np.amin(data.FLUX[validWavelengthIndices]) - np.amax(data.ERROR[validWavelengthIndices]))

        return (np.amin(minFluxes), np.amax(maxFluxes))

    def computeFluxForAbMagitude(self, abMagnitude, grism):
        # erg s−1 cm−2 Hz−1

        perHzUnit = astrounits.erg / astrounits.second / astrounits.Hz / astrounits.cm**2
        fluxPerHz = 10**(-2.0 * (abMagnitude + 48.6) / 5.0) * perHzUnit
        perAnstromUnit = astrounits.erg / astrounits.second / astrounits.angstrom / astrounits.cm**2
        fluxPerAngstrom = fluxPerHz.to(perAnstromUnit,
                                       equivalencies=astrounits.spectral_density(
                                           SpecPlotter.filterPivotWavelengthsForGrism[grism][1].to(astrounits.angstrom))
                                       )
        return fluxPerAngstrom

    def plotContamination(self, baseAxes):
        for grism, specRange in SpecPlotter.spectrumRanges.items():
            if not isinstance(grism, int):
                continue
            grismRange = specRange.to(astrounits.angstrom).value
            fullRange = SpecPlotter.plottedWavelengthRange.to(astrounits.angstrom).value

            validWavelengths = ((self.spectralData[grism].WAVELENGTH.diff() > 0) &
                                (self.spectralData[grism].WAVELENGTH < grismRange[1]) &
                                (self.spectralData[grism].WAVELENGTH > grismRange[0]) &
                                (self.spectralData[grism].WAVELENGTH < fullRange[1]) &
                                (self.spectralData[grism].WAVELENGTH > fullRange[0]))

            mplplot.fill_between(self.spectralData[grism].WAVELENGTH[validWavelengths],
                                 0,
                                 self.spectralData[grism].CONTAM[validWavelengths],
                                 color=SpecPlotter.spectrumColours[grism],
                                 alpha=0.2,
                                 label='G{} Contamination'.format(grism))
        return baseAxes

    def plotAbMagnitudesAsFluxes(self, abMagnitudes, targetAxes):
        for grism, abMagnitude in abMagnitudes.items():
            if not np.isnan(abMagnitude):
                fluxForMagnitude = self.computeFluxForAbMagitude(
                    abMagnitude=abMagnitude, grism=grism)
                lowError = SpecPlotter.filterPivotWavelengthsForGrism[grism][
                    1] - SpecPlotter.filterWavelengthRangesForGrism[grism][1][0]
                highError = SpecPlotter.filterWavelengthRangesForGrism[grism][
                    1][1] - SpecPlotter.filterPivotWavelengthsForGrism[grism][1]
                errorBars = mplplot.errorbar(SpecPlotter.filterPivotWavelengthsForGrism[grism][1].to(astrounits.angstrom).value,
                                             fluxForMagnitude.value,
                                             axes=targetAxes,
                                             marker='o',
                                             xerr=np.array([[lowError.value], [highError.value]]),
                                             c=SpecPlotter.spectrumColours[grism],
                                             ms=10,
                                             elinewidth=2,
                                             capthick=2,
                                             label='$f\,(m_{}^{})$'.format(r'{\rm AB}', r'{{\rm F' + str(SpecPlotter.filterPivotWavelengthsForGrism[grism][0]) + 'W}}'))
                errorBars[-1][0].set_linestyle('--')

    def plotZerothOrders(self, targetAxes):
        zerothOrderData = self.spectralData['COMBINED']['ZEROTH']
        indicesWithZerothOrders = np.where(zerothOrderData > 1)
        groupedIndicesWithZerothOrders = [list(group) for _, group in itertools.groupby(
            indicesWithZerothOrders[0], key=lambda n, c=itertools.count():n - next(c))]
        for group in groupedIndicesWithZerothOrders:
            print(self.spectralData['COMBINED']['WAVELENGTH'][group[0]],
                  self.spectralData['COMBINED']['WAVELENGTH'][group[-1]])
            currentAxis = mplplot.gca()
            mplplot.sca(targetAxes)
            mplplot.axvspan(self.spectralData['COMBINED']['WAVELENGTH'][group[0]],
                            self.spectralData['COMBINED']['WAVELENGTH'][group[-1]],
                            fc='gray', ec='gray', fill=True, alpha=0.3)
            mplplot.sca(currentAxis)

    def plotSpectrum(self, savePath=None, abMagnitudes=None, gridSpec=None):
        plotAxes = None if gridSpec is None else mplplot.subplot(gridSpec)
        plotYLimits = self.computePlottingLimits()
        plotAxes.set_ylabel(r'Observed Flux (erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)', size='large')
        # plotAxes.set_ylim(0, plotYLimits[1])
        for grism, data in self.spectralData.items():
            if data is not None:
                grismRange = SpecPlotter.spectrumRanges[grism].to(astrounits.angstrom).value
                fullRange = SpecPlotter.plottedWavelengthRange.to(astrounits.angstrom).value

                runningMaxWavelength = np.maximum.accumulate(data.WAVELENGTH)
                _, increasingWavelengthIndices = np.unique(runningMaxWavelength, return_index=True)

                increasingWavelengthData = data.iloc[increasingWavelengthIndices]

                wavelengthCut = ((increasingWavelengthData.WAVELENGTH < grismRange[1]) &
                                 (increasingWavelengthData.WAVELENGTH > grismRange[0]) &
                                 (increasingWavelengthData.WAVELENGTH < fullRange[1]) &
                                 (increasingWavelengthData.WAVELENGTH > fullRange[0]))

                plotData = increasingWavelengthData[wavelengthCut]
                if len(plotData.index) == 0:
                    continue

                plotAxes = plotData.plot.line(ax=plotAxes,
                                              x='WAVELENGTH',
                                              y='FLUX',
                                              yerr=('ERROR' if not isinstance(
                                                  grism, int) else None),
                                              logy=False,
                                              c=SpecPlotter.spectrumColours[grism],
                                              alpha=0.6,
                                              label='{} Spectrum'.format(
                                                  SpecPlotter.spectrumLabels[grism]),
                                              linewidth=(1 if not isinstance(grism, int) else 2),
                                              sharex=True
                                              )

        if abMagnitudes is not None:
            self.plotAbMagnitudesAsFluxes(abMagnitudes=abMagnitudes, targetAxes=plotAxes)

        plotAxes = self.plotContamination(plotAxes)

        self.plotZerothOrders(plotAxes)

        handles, labels = plotAxes.get_legend_handles_labels()
        plotAxes.legend(handles, labels, fontsize='large', handlelength=5, ncol=1, loc='best')

        # axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)

        plotAxes.set_xlabel(r'Observed Wavelength (${\rm {\AA}}$)', size='large')
        plotAxes.set_title('Field {}, Object {}: 1D Spectra'.format(self.targetPar,
                                                                    self.targetObject))
        plotAxes.set_xlim(*SpecPlotter.plottedWavelengthRange.value)

        if savePath is not None:
            mplplot.savefig(savePath, dpi=300, bbox_inches='tight')
            mplplot.close()
