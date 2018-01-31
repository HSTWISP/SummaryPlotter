import pandas as pd
import numpy as np
import scipy as sp
import scipy.interpolate
import astropy.units as astrounits
import os
import itertools
import matplotlib.pyplot as mplplot
import matplotlib.gridspec as mplgs
import glob
import re

from SpecLoader import SpecLoader


# TODO: Refactor to use SpecLoader


class SpecPlotter:

    legendProperties = {'fontsize': 'large',
                        'handlelength': 5,
                        'ncol': 2,
                        'loc': 'upper center'}

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

    spectrumMarkers = {
        141: None,
        102: None,
        'COMBINED': None,
        'MODEL': '|'}

    spectrumRanges = {
        # (1075, 1700)*astrounits.nanometer,
        141: (1100, 1700) * astrounits.nanometer,
        # (800, 1150)*astrounits.nanometer
        102: (800, 1150) * astrounits.nanometer,
        'COMBINED': (800, 1700) * astrounits.nanometer,
        'MODEL': (800, 1700) * astrounits.nanometer}

    filterPivotWavelengthsForGrism = {102: (110, 11534.46 * astrounits.angstrom),
                                      141: (160, 15370.33 * astrounits.angstrom)}

    filterWavelengthRangesForGrism = {102: (110, (8832, 14121) * astrounits.angstrom),
                                      141: (160, (13854, 16999) * astrounits.angstrom)}

    plottedWavelengthRange = (8500, 16500) * astrounits.angstrom

    def __init__(self,
                 withGrismSpectrumPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/Spectra/Par{par}_G{grism}_BEAM_{object}A.dat',
                 bothGrismSpectrumPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/Spectra/Par{par}_BEAM_{object}A.dat',
                 modelSpectrumPathPattern='/Volumes/ramon2_wisps/z_estimates/bc03_models/{model_name}',
                 verboseOutput=False):
        self.reset(withGrismSpectrumPathPattern,
                   bothGrismSpectrumPathPattern,
                   modelSpectrumPathPattern,
                   verboseOutput)

    def reset(self,
              withGrismSpectrumPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/Spectra/Par{par}_G{grism}_BEAM_{object}A.dat',
              bothGrismSpectrumPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/Spectra/Par{par}_BEAM_{object}A.dat',
              modelSpectrumPathPattern='/Volumes/ramon2_wisps/z_estimates/bc03_models/{model_name}',
              verboseOutput=False):
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
        self.legendAxes = None

        self.verboseOutput = verboseOutput

    def loadSpectralData(self, targetObject, targetPar, bestFitModelName=None, bestFitModelNorm=None, bestFitRedshift=0, subtractContamination=True, pipelineVersion=6.2):
        # lazy assignment of targetPar and pipelineVersion
        self.targetObject = targetObject
        self.targetPar = targetPar
        self.bestFitModelName = bestFitModelName
        self.bestFitModelNorm = bestFitModelNorm
        self.subtractContamination = subtractContamination
        self.bestFitRedshift = bestFitRedshift
        self.pipelineVersion = pipelineVersion
        # load spectral data for individual grisms
        for grism in self.spectralData.keys():
            testSpectrumPath = self.spectrumPathPatterns[grism].format(
                pipeline_version=self.pipelineVersion,
                par=self.targetPar,
                grism=grism,
                object=targetObject,
                model_name=self.bestFitModelName)

            if os.path.isfile(testSpectrumPath) and os.access(testSpectrumPath, os.R_OK):
                if isinstance(grism, str) and 'MODEL' in grism and self.bestFitModelName is not None and self.bestFitModelNorm is not None:
                    self.spectralData[grism] = pd.read_csv(
                        testSpectrumPath,
                        header=None,
                        comment='#',
                        delim_whitespace=True,
                        names=['WAVELENGTH', 'FLUX']).dropna()
                    self.spectralData[grism].FLUX /= self.bestFitModelNorm
                    self.spectralData[
                        grism].WAVELENGTH *= (1.0 + self.bestFitRedshift)
                    self.spectralData[grism]['ERROR'] = np.zeros_like(
                        self.spectralData[grism].FLUX)
                else:
                    self.spectralData[grism] = pd.read_csv(
                        testSpectrumPath,
                        skipinitialspace=True,
                        header=None,
                        comment='#',
                        delim_whitespace=True,
                        names=['WAVELENGTH', 'FLUX', 'ERROR', 'CONTAM', 'ZEROTH']).dropna()
                    if self.subtractContamination:
                        self.spectralData[
                            grism].FLUX -= self.spectralData[grism].CONTAM

    def computePlottingLimits(self):
        minFluxes = []
        maxFluxes = []

        for grism, data in self.spectralData.items():
            if data is not None:
                grismRange = SpecPlotter.spectrumRanges[
                    grism].to(astrounits.angstrom).value
                fullRange = SpecPlotter.plottedWavelengthRange.to(
                    astrounits.angstrom).value
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
        perAnstromUnit = astrounits.erg / astrounits.second / \
            astrounits.angstrom / astrounits.cm**2
        fluxPerAngstrom = fluxPerHz.to(perAnstromUnit,
                                       equivalencies=astrounits.spectral_density(
                                           SpecPlotter.filterPivotWavelengthsForGrism[grism][1].to(astrounits.angstrom))
                                       )
        return fluxPerAngstrom

    def plotContamination(self, baseAxes):
        for grism, specRange in SpecPlotter.spectrumRanges.items():
            if not isinstance(grism, int) or self.spectralData[grism] is None:
                continue
            grismRange = specRange.to(astrounits.angstrom).value
            fullRange = SpecPlotter.plottedWavelengthRange.to(
                astrounits.angstrom).value

            validWavelengths = ((self.spectralData[grism].WAVELENGTH.diff() > 0) &
                                (self.spectralData[grism].WAVELENGTH < grismRange[1]) &
                                (self.spectralData[grism].WAVELENGTH > grismRange[0]) &
                                (self.spectralData[grism].WAVELENGTH < fullRange[1]) &
                                (self.spectralData[grism].WAVELENGTH > fullRange[0]))

            mplplot.fill_between(self.spectralData[grism].WAVELENGTH[validWavelengths],
                                 0,
                                 self.spectralData[grism].CONTAM[
                                     validWavelengths],
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
                                             xerr=np.array(
                                                 [[lowError.value], [highError.value]]),
                                             c=SpecPlotter.spectrumColours[
                                                 grism],
                                             ms=10,
                                             elinewidth=2,
                                             capthick=2,
                                             label='$f\,(m_{}^{})$'.format(r'{\rm AB}', r'{{\rm F' + str(SpecPlotter.filterPivotWavelengthsForGrism[grism][0]) + 'W}}'))
                errorBars[-1][0].set_linestyle('--')

    def matchSpectrumNormalizations(self, targetSpectrum, fiducialSpectrum):
        """
        Computes the multiplicative scaling that must be applied to targetSpectrum
        in order to minimize the residuals between it and fiducialSpectrum
        """
        normFactor = np.sum(
            targetSpectrum.FLUX * fiducialSpectrum.FLUX) / np.sum(targetSpectrum.FLUX**2)
        return normFactor

    def interpolateModelSpectrum(self, rawModelSpectrum, samplingExampleSpectrum):
        # Interpolate the values of the Y axis to enable simple resampling of the modelData
        # at the data sampling points.
        dataInterpolator = sp.interpolate.InterpolatedUnivariateSpline(x=rawModelSpectrum.WAVELENGTH,
                                                                       y=rawModelSpectrum.FLUX,
                                                                       k=1)

        interpolatedSpectrum = samplingExampleSpectrum.copy()
        interpolatedSpectrum.FLUX = dataInterpolator(
            interpolatedSpectrum.WAVELENGTH.values)
        interpolatedSpectrum.ERROR = np.zeros_like(interpolatedSpectrum.ERROR)

        return interpolatedSpectrum

    def resampleModelSpectrum(self, rawModelSpectrum, samplingExampleSpectrum):
        # Interpolate the values of the Y axis to enable rebinning by integration
        # between new bounds
        dataInterpolator = sp.interpolate.InterpolatedUnivariateSpline(rawModelSpectrum.WAVELENGTH,
                                                                       rawModelSpectrum.FLUX,
                                                                       k=1)

        binBoundaries = [
            *samplingExampleSpectrum.head(1).WAVELENGTH.values,
            *(0.5 * (samplingExampleSpectrum.WAVELENGTH.iloc[
              1:].values + samplingExampleSpectrum.WAVELENGTH.iloc[:-1].values)),
            *samplingExampleSpectrum.tail(1).WAVELENGTH.values]

        resampledFluxes = [dataInterpolator.integral(minWavelength, maxWavelength) / (
            maxWavelength - minWavelength) for (minWavelength, maxWavelength) in zip(binBoundaries[:-1], binBoundaries[1:])]
        resampledSpectrum = samplingExampleSpectrum.copy()
        resampledSpectrum.FLUX = resampledFluxes
        resampledSpectrum.ERROR = np.zeros_like(resampledSpectrum.ERROR)
        return resampledSpectrum

    def plotZerothOrders(self, targetAxes):
        zerothOrderData = self.spectralData['COMBINED']['ZEROTH']
        indicesWithZerothOrders = np.where(zerothOrderData > 1)
        groupedIndicesWithZerothOrders = [list(group) for _, group in itertools.groupby(
            indicesWithZerothOrders[0], key=lambda n, c=itertools.count():n - next(c))]

        for group in groupedIndicesWithZerothOrders:
            print(self.spectralData['COMBINED']['WAVELENGTH'].iloc[group[0]],
                  self.spectralData['COMBINED']['WAVELENGTH'].iloc[group[-1]])
            currentAxis = mplplot.gca()
            mplplot.sca(targetAxes)
            mplplot.axvspan(self.spectralData['COMBINED']['WAVELENGTH'].iloc[group[0]],
                            self.spectralData['COMBINED'][
                                'WAVELENGTH'].iloc[group[-1]],
                            fc='gray', ec='gray', fill=True, alpha=0.3)
            mplplot.sca(currentAxis)

    def plotSpectrum(self,
                     savePath=None,
                     abMagnitudes=None,
                     gridSpec=None,
                     titlePrefix='',
                     titleSuffix='',
                     legendProperties=None,
                     legendGridSpec=None):

        plotLegendProperties = SpecPlotter.legendProperties
        if legendProperties is not None:
            plotLegendProperties.update(legendProperties)
        self.legendAxes = None if legendGridSpec is None else mplplot.subplot(
            legendGridSpec)

        plotAxes = None if gridSpec is None else mplplot.subplot(gridSpec)
        plotYLimits = self.computePlottingLimits()
        plotAxes.set_ylabel(
            r'Observed Flux (erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)', size='large')
        # plotAxes.set_ylim(0, plotYLimits[1])

        if self.spectralData['MODEL'] is not None:
            self.bestModelNormFactor = 1
            self.spectralData['MODEL'] = self.interpolateModelSpectrum(
                self.spectralData['MODEL'], self.spectralData['COMBINED'])

        for grism, data in self.spectralData.items():
            if data is not None:
                grismRange = SpecPlotter.spectrumRanges[
                    grism].to(astrounits.angstrom).value
                fullRange = SpecPlotter.plottedWavelengthRange.to(
                    astrounits.angstrom).value

                runningMaxWavelength = np.maximum.accumulate(data.WAVELENGTH)
                _, increasingWavelengthIndices = np.unique(
                    runningMaxWavelength, return_index=True)

                increasingWavelengthData = data.iloc[
                    increasingWavelengthIndices]

                wavelengthCut = ((increasingWavelengthData.WAVELENGTH < grismRange[1]) &
                                 (increasingWavelengthData.WAVELENGTH > grismRange[0]) &
                                 (increasingWavelengthData.WAVELENGTH < fullRange[1]) &
                                 (increasingWavelengthData.WAVELENGTH > fullRange[0]))

                plotData = increasingWavelengthData[wavelengthCut]
                if len(plotData.index) == 0:
                    continue

                if grism == 'COMBINED':
                    nanLocIndicator = (plotData.ERROR / plotData.FLUX > 10)
                    plotData.loc[nanLocIndicator, 'FLUX'] = np.nan

                plotAxes = plotData.plot.line(ax=plotAxes,
                                              x='WAVELENGTH',
                                              y='FLUX',
                                              yerr=('ERROR' if not isinstance(
                                                  grism, int) else None),
                                              logy=False,
                                              c=SpecPlotter.spectrumColours[
                                                  grism],
                                              alpha=0.6,
                                              label='{} Spectrum'.format(
                                                  SpecPlotter.spectrumLabels[grism]),
                                              linewidth=(1 if not isinstance(
                                                  grism, int) else 2),
                                              marker=SpecPlotter.spectrumMarkers[
                                                  grism],
                                              sharex=False
                                              )

        if abMagnitudes is not None:
            self.plotAbMagnitudesAsFluxes(
                abMagnitudes=abMagnitudes, targetAxes=plotAxes)

        if not self.subtractContamination:
            plotAxes = self.plotContamination(plotAxes)

        self.plotZerothOrders(plotAxes)

        handles, labels = plotAxes.get_legend_handles_labels()
        if self.legendAxes is not None:
            self.legendAxes.legend(handles, labels, **plotLegendProperties)
            self.legendAxes.set_axis_off()
            plotAxes.legend().remove()
        else:
            self.legendAxes = plotAxes
            plotAxes.legend(handles, labels, **plotLegendProperties)

        # axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)

        plotAxes.set_xlabel(
            r'Observed Wavelength (${\rm {\AA}}$)', size='large')
        plotAxes.set_title('{}Field {}, Object {}: 1D Spectra{}'.format(titlePrefix, self.targetPar,
                                                                        self.targetObject, titleSuffix))

        plotAxes.set_xlim(*SpecPlotter.plottedWavelengthRange.value)

        if savePath is not None:
            mplplot.savefig(savePath, dpi=300, bbox_inches='tight')
            mplplot.close()
        else:
            return plotAxes

    def makeStandardGridSpec(self):
        # generate a standard grid specification that includes positions for
        # the stamp imagegridSpec
        fullGridSpec = mplgs.GridSpec(
            3, 2, width_ratios=[4, 1], height_ratios=[4, 1, 1])
        # return the portion of the grid specification into which the spectra
        # should be plotted.
        return fullGridSpec[0, 0]
