import os
import warnings

import astropy.io.fits as astrofits
import astropy.units as astrounits
import astropy.wcs as astrowcs
import astropy.visualization as astrovis
import astropy.visualization.mpl_normalize as astromplnorm
import astropy.nddata as astronddata
import astropy.nddata.utils as astrondutils

import astropy.units as astrounits

import matplotlib
import matplotlib.pyplot as mplplot
import matplotlib.patches as mplpatches
import matplotlib.gridspec as mplgs
import numpy as np
import scipy as sp
import scipy.stats as spstats

import pyregion
import copy


class StampPlotter:
    """
    Class that plots 2D drizzled stamps produced by the WISP pipeline and also
    extracts and plots 'postage stamp' cutouts from the corresponding direct image.
    """

    grismRanges = {
        141: (1075, 1700) * astrounits.nanometer,
        102: (800, 1150) * astrounits.nanometer}

    # deliberately favour F140 over F160 for G141
    grismToFilterMap = {102: [110], 141: [140, 160]}

    plottedWavelengthRange = (8500, 16500) * astrounits.angstrom

    def __init__(self,
                 drizzledStampPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/G{grism}_DRIZZLE/aXeWFC3_G{grism}_mef_ID{object}.fits',
                 directImagePathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/DATA/DIRECT_GRISM/F{filter}.fits',
                 cutoutSizePixels=40,
                 doTrimBorderPixels=True,
                 verboseOutput=False):
        self.reset(drizzledStampPathPattern, directImagePathPattern,
                   cutoutSizePixels, doTrimBorderPixels, verboseOutput)

    def reset(self,
              drizzledStampPathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/G{grism}_DRIZZLE/aXeWFC3_G{grism}_mef_ID{object}.fits',
              directImagePathPattern='/Volumes/ramon2_wisps/data/V{pipeline_version}/Par{par}/DATA/DIRECT_GRISM/F{filter}.fits',
              cutoutSizePixels=40,
              doTrimBorderPixels=True,
              verboseOutput=False):
        self.drizzledStampPathPattern = drizzledStampPathPattern
        self.directImagePathPattern = directImagePathPattern
        self.cutoutSizePixels = cutoutSizePixels
        self._doTrimBorderPixels = doTrimBorderPixels
        self._verboseOutput = verboseOutput

        self.targetObject = None
        self.targetPar = None
        self.pipelineVersion = None
        self.stampPaths = None
        self.stampHdus = {'SCI': None, 'WHT': None, 'MOD': None, 'CON': None}
        self.directPaths = None
        self.directHdus = None
        self.directCutouts = None
        self.directCutoutExtentsArcsec = {
            grism: None for grism in StampPlotter.grismToFilterMap.keys()}
        self._stretchModel = None
        self._stretchInterval = None
        self._useOptimalZScaleForStamp = None
        self._useOptimalZScaleForDirectCutout = None

    @property
    def verboseOutput(self):
        return self._verboseOutput

    @verboseOutput.setter
    def verboseOutput(self, verbose):
        self._verboseOutput = verbose
        if self.verboseOutput:
            print('Verbose output activated.')

    @property
    def doTrimBorderPixels(self):
        if self.verboseOutput:
            print('Border pixels{}will be trimmed.'.format(
                ' ' if self._doTrimBorderPixels else ' not '))
        return self._doTrimBorderPixels

    @doTrimBorderPixels.setter
    def doTrimBorderPixels(self, doTrimBorderPixels):
        self._doTrimBorderPixels = doTrimBorderPixels

    @property
    def stretchInterval(self):
        if self._stretchInterval is None:
            print('Using default MinMaxInterval stretch interval')
            return astrovis.MinMaxInterval()
        return self._stretchInterval

    @stretchInterval.setter
    def stretchInterval(self, stretchInterval):
        self._stretchInterval = stretchInterval

    @property
    def stretchModel(self):
        if self._stretchModel is None:
            print('Using default HistEqStretch stretch model')
            return lambda target, header, data: astrovis.HistEqStretch(data)
        return self._stretchModel

    @stretchModel.setter
    def stretchModel(self, stretchModel):
        self._stretchModel = stretchModel

    @property
    def useOptimalZScaleForStamp(self):
        return self._useOptimalZScaleForStamp

    @useOptimalZScaleForStamp.setter
    def useOptimalZScaleForStamp(self, useOptimalZScale=True):
        self._useOptimalZScaleForStamp = useOptimalZScale

    @property
    def useOptimalZScaleForDirectCutout(self):
        return self._useOptimalZScaleForDirectCutout

    @useOptimalZScaleForDirectCutout.setter
    def useOptimalZScaleForDirectCutout(self, useOptimalZScale=True):
        self._useOptimalZScaleForDirectCutout = useOptimalZScale

    def getDirectFilterForGrism(self, grism):
        if grism in self.grismToFilterMap:
            for directFilterNumber in self.grismToFilterMap[grism]:
                testPath = self.directImagePathPattern.format(
                    pipeline_version=self.pipelineVersion,
                    par=self.targetPar,
                    filter=directFilterNumber)
                if os.path.isfile(testPath):
                    return directFilterNumber
        return None

    def interpolateDirectCutout(self, directData, stampHeader):
        halfCutoutSizePixels = np.floor_divide(self.cutoutSizePixels, 2)
        refY = np.floor(stampHeader['REFPNTY'])
        refX = np.floor(stampHeader['REFPNTX'])
        cutoutLimits = (
            (
                int(max(0, refY - halfCutoutSizePixels)),
                int(min(refY + halfCutoutSizePixels, directData.shape[0]))
            ),
            (
                int(max(0, refX - halfCutoutSizePixels)),
                int(min(refX + halfCutoutSizePixels, directData.shape[1]))
            ))
        return directImageData[cutoutLimits[0][0]: cutoutLimits[0][1], cutoutLimits[1][0]: cutoutLimits[1][1]]

    def extractDirectCutout(self, directData, stampHeader):
        halfCutoutSizePixels = np.floor_divide(self.cutoutSizePixels, 2)
        refY = np.ceil(stampHeader['REFPNTY'])
        refX = np.ceil(stampHeader['REFPNTX'])
        cutoutLimits = (
            (
                int(max(0, refY - halfCutoutSizePixels)),
                int(min(refY + halfCutoutSizePixels, directData.shape[0]))
            ),
            (
                int(max(0, refX - halfCutoutSizePixels)),
                int(min(refX + halfCutoutSizePixels, directData.shape[1]))
            ))
        return directData[cutoutLimits[0][0]: cutoutLimits[0][1], cutoutLimits[1][0]: cutoutLimits[1][1]]

    def loadDrizzledStamps(self, targetObject, targetPar, pipelineVersion=6.2):
        # lazy assignment of pipelineVersion, targetPar and targetObject
        self.targetPar = targetPar
        self.targetObject = targetObject
        self.pipelineVersion = pipelineVersion
        self.stampPaths = {
            102: self.drizzledStampPathPattern.format(
                pipeline_version=self.pipelineVersion,
                par=self.targetPar,
                grism=102,
                object=self.targetObject),
            141: self.drizzledStampPathPattern.format(
                pipeline_version=self.pipelineVersion,
                par=self.targetPar,
                grism=141,
                object=self.targetObject)
        }

        for extName in self.stampHdus.keys():
            self.stampHdus[extName] = {key: (astrofits.getdata(stampPath, ext=(extName, 1), header=True) if os.path.isfile(
                stampPath) else None) for key, stampPath in self.stampPaths.items()}

    def loadDirectCutouts(self, targetObject, targetPar, pipelineVersion=6.2):
        if self.stampHdus is None:
            self.loadDrizzledStamps(targetObject, targetPar, pipelineVersion)

        self.directPaths = {
            102: self.directImagePathPattern.format(
                pipeline_version=self.pipelineVersion,
                par=self.targetPar,
                filter=self.getDirectFilterForGrism(102)),
            141: self.directImagePathPattern.format(
                pipeline_version=self.pipelineVersion,
                par=self.targetPar,
                filter=self.getDirectFilterForGrism(141))
        }

        self.directHdus = {key: (astrofits.getdata(directFilePath, ext=('PRIMARY', 1), header=True) if os.path.isfile(
            directFilePath) else None) for key, directFilePath in self.directPaths.items()}
        self.directCutouts = {key: (self.extractDirectCutout(self.directHdus[key][0], self.stampHdus['SCI'][key][1]) if (self.stampHdus[
                                    'SCI'][key] is not None and self.directHdus[key] is not None) else None) for key, directHdu in self.directHdus.items()}

    def getExtractionRange(self, modelData, axis):
        # -- Compute the total sum of model pixel values
        modelSum = np.sum(modelData)
        # -- Compute the projected sum of model pixel values along the specifies axis
        modelProjection = np.sum(modelData, axis=axis, keepdims=False)
        # -- Compute the CDF of projected model pixel value sums
        modelCdf = np.cumsum(modelProjection) / modelSum
        # -- Finally extract the pixel indices in the Y-direction for which the
        #    CDF falls between the desired bounds
        rangePixels = np.argwhere((modelCdf > 0.01) & ~(modelCdf > 0.99))
        # -- Increment the first and last values of the extracted indices and
        #    use these as the extraction bounds
        rangePixelBounds = (rangePixels[0][0] + 1, rangePixels[-1][0] + 1)
        return (rangePixelBounds, rangePixels)

    def getTrimmedStampData(self, data, modelData, contamData, grism, wcs=None):
        # - Determine appropriate range for final image from MOD extension
        # modelData = self.stampHdus['MOD'][0]
        yBounds, pixelsInYBounds = self.getExtractionRange(modelData, 1)
        xBounds, pixelsInXBounds = self.getExtractionRange(modelData, 0)

        yMidPoint = 0.5 * (yBounds[0] + yBounds[1])
        yExtent = 5 * (yBounds[1] - yBounds[0])

        yRange = (int(np.floor(yMidPoint - 0.5 * yExtent)),
                  int(np.ceil(yMidPoint + 0.5 * yExtent)))

        trimmedData = astrondutils.Cutout2D(data=data,
                                            wcs=wcs,
                                            position=(
                                                0.5 * (xBounds[0] + xBounds[1]), yMidPoint),
                                            size=(yExtent, xBounds[
                                                  1] - xBounds[0]),
                                            mode='trim')

        stampHeader = self.stampHdus['SCI'][grism][1]

        modelRectangle = mplpatches.Rectangle(xy=(xBounds[0] - stampHeader['CRPIX2'],
                                                  yBounds[0] - stampHeader['CRPIX1'] + 0.5 * (yBounds[1] - yBounds[0])),
                                              height=1.5 *
                                              (yBounds[1] - yBounds[0]),
                                              width=xBounds[1] - xBounds[0])
        modelRectangle.set_ec('white')
        modelRectangle.set_fill(False)
        modelRectangle.set_linewidth(2)
        modelRectangle.set_linestyle('--')

        if self.verboseOutput:
            print(trimmedData.shape, data.shape)

        return trimmedData, modelData[yRange[0]:yRange[1], xBounds[0]:xBounds[1]], contamData[yRange[0]:yRange[1], xBounds[0]:xBounds[1]], modelRectangle

    def buildWCSObject(self, stampHeader):
        # set up WCS object
        wcsObject = astrowcs.WCS(stampHeader)
        # hack to undo default behaviour of transformation of wavelength to
        # metres
        wcsObject.wcs.cdelt[0] *= 1e10
        wcsObject.wcs.crval[0] *= 1e10
        return wcsObject

    def processRawZerothOrders(self, rawZerothOrders, grism, wcsObject):
        # operate on a copy of the parsed region data
        zerothOrders = copy.deepcopy(rawZerothOrders[grism])
        processedZerothOrders = []
        processedZerothOrdersComments = []

        drizzledStampScienceHeader = self.stampHdus['SCI'][grism][1]
        print(repr(drizzledStampScienceHeader))
        for region in zerothOrders:
            stampReferencePoint = (drizzledStampScienceHeader[
                                   'RREFPNTX'], drizzledStampScienceHeader['RREFPNTY'])
            imageReferencePoint = (drizzledStampScienceHeader[
                                   'REFPNTX'], drizzledStampScienceHeader['REFPNTY'])
            drizzledStampBBoxCoordinates = ((drizzledStampScienceHeader['BB0X'], drizzledStampScienceHeader['BB0Y']),
                                            (drizzledStampScienceHeader['BB1X'], drizzledStampScienceHeader['BB1Y']))
            # At this stage, discard any regions that do not fall within the
            # stamp
            if (region.coord_list[0] > drizzledStampBBoxCoordinates[0][0]
                    and region.coord_list[0] < drizzledStampBBoxCoordinates[1][0]
                    and region.coord_list[1] > drizzledStampBBoxCoordinates[0][1]
                    and region.coord_list[1] < drizzledStampBBoxCoordinates[1][1]):
                print('Start: {}, {}, {}'.format(region.coord_list,
                                                 drizzledStampBBoxCoordinates, imageReferencePoint))
                # Convert from non-drizzled grism image coordinates into stamp bounding
                # box coordinates
                # drizzledStampBBoxCoordinates[0][0]
                region.coord_list[0] -= imageReferencePoint[0]
                # drizzledStampBBoxCoordinates[0][1]
                region.coord_list[1] -= imageReferencePoint[1]
                print('BBOX: {}'.format(region.coord_list))
                # Apply second order corrections for distortions introduced by the drizzling process
                # Apply a 1 pixel decrement because ds9 region files enumerate pixels from 1
                # whereas the drizzle correction assumes a zero pixel origin
                # region.coord_list[0] -= 1
                # region.coord_list[1] -= 1

                # Apply the drizzle corrections as specified in:
                # http://adsabs.harvard.edu/abs/2005ASPC..347..138K
                correctedX = drizzledStampScienceHeader['D001OUXC'] + (drizzledStampScienceHeader['DRZ00'] + drizzledStampScienceHeader['DRZ01'] * (region.coord_list[
                                                                       0] - drizzledStampScienceHeader['D001INXC']) + drizzledStampScienceHeader['DRZ02'] * (region.coord_list[1] - drizzledStampScienceHeader['D001INYC']))
                correctedY = drizzledStampScienceHeader['D001OUYC'] + (drizzledStampScienceHeader['DRZ10'] + drizzledStampScienceHeader['DRZ11'] * (region.coord_list[
                                                                       0] - drizzledStampScienceHeader['D001INXC']) + drizzledStampScienceHeader['DRZ12'] * (region.coord_list[1] - drizzledStampScienceHeader['D001INYC']))

                # There is no need to reapply the 1 pixel offset since this is handled by the wcs_pix2world call
                # processing that happens later
                region.coord_list[0] = correctedX
                region.coord_list[1] = correctedY

                region.coord_list[0] += stampReferencePoint[0]
                region.coord_list[1] += stampReferencePoint[1]

                # worldCoords = wcsObject.wcs_pix2world(region.coord_list[0], region.coord_list[1], 0)
                # print(worldCoords[0], worldCoords[1])
                # region.coord_list[0] = np.asscalar(worldCoords[0])
                # region.coord_list[1] = np.asscalar(worldCoords[1])

                print('DRIZZLE: {}'.format(region.coord_list))

                processedZerothOrders.append(copy.deepcopy(region))
                processedZerothOrdersComments.append('')

        # Convert from plain list to ShapeList
        processedZerothOrders = pyregion.ShapeList(
            processedZerothOrders, processedZerothOrdersComments)
        del zerothOrders
        return processedZerothOrders

    def plotZerothOrders(self, zerothOrderData, targetAxes, grism, wcsObject):
        regionGraphics = self.processRawZerothOrders(
            zerothOrderData, grism, wcsObject).get_mpl_patches_texts()
        for regionGraphic in regionGraphics[0]:
            regionGraphic.set_edgecolor('r')
            # regionGraphic.set_transform(targetAxes.transData)
            targetAxes.add_patch(regionGraphic)

    def plotDrizzledStamps(self, extName='SCI', colourMap='viridis', applyWCS=True, savePath=None, gridSpecs=None, zerothOrderData=None, titlePrefix='', titleSuffix=''):
        if self.stampHdus[extName] is not None:
            allSubPlotAxes = []
            if gridSpecs is None:
                mplplot.figure(figsize=(10, 5))
            for stampIndex, (grism, hduData) in enumerate(self.stampHdus[extName].items()):
                if hduData is None:
                    if gridSpecs is None:
                        subplotAxes = mplplot.subplot(2, 1, stampIndex + 1)
                    else:
                        subplotAxes = mplplot.subplot(gridSpecs[stampIndex])
                    allSubPlotAxes.append(subplotAxes)
                    subplotAxes.text(0.5,
                                     0.5,
                                     'Field {}, Object {}:\nNO DATA AVAILABLE.'.format(self.targetPar,
                                                                                       self.targetObject),
                                     horizontalalignment='center',
                                     fontsize='large',
                                     transform=subplotAxes.transAxes)
                    continue
                if np.all(hduData[0] < 0):
                    if gridSpecs is None:
                        subplotAxes = mplplot.subplot(2, 1, stampIndex + 1)
                    else:
                        subplotAxes = mplplot.subplot(gridSpecs[stampIndex])
                    subplotAxes.text(0.5,
                                     0.5,
                                     'Field {}, Object {}:\nNO NONZERO DATA AVAILABLE.'.format(self.targetPar,
                                                                                               self.targetObject),
                                     horizontalalignment='center',
                                     fontsize='large',
                                     transform=subplotAxes.transAxes)
                    continue
                stampHeader = hduData[1]
                stampData = hduData[0]

                wcsObject = None

                if applyWCS:
                    wcsObject = self.buildWCSObject(stampHeader)
                    wavelengthUnit = r'${\rm {\AA}}$'
                    # wavelengthUnit = 'm'
                    # leave X-dispersion in arsecond units
                    xDispUnit = wcsObject.wcs.cunit[1]
                else:
                    wavelengthUnit = xDispUnit = 'Pixel'
                # construct a normalization handler for the image

                cutoutData = stampData
                modelData = self.stampHdus['MOD'][grism][0]
                cutoutModel = modelData
                contamData = self.stampHdus['CON'][grism][0]
                cutoutContam = contamData
                modelRect = None
                if self.doTrimBorderPixels:
                    cutoutData, cutoutModel, cutoutContam, modelRect = self.getTrimmedStampData(
                        stampData, modelData, contamData, grism, wcs=wcsObject)
                    stampData = cutoutData.data
                    wcsObject = cutoutData.wcs

                if self.useOptimalZScaleForStamp:
                    (vMin, vMax), interval = self.computeDefaultZRange(
                        stampData, cutoutContam)
                    norm = astromplnorm.ImageNormalize(stampData,
                                                       interval=interval,
                                                       stretch=self.stretchModel(
                                                           self, None, stampData[cutoutModel > 0]))
                else:
                    norm = astromplnorm.ImageNormalize(stampData,
                                                       interval=self.stretchInterval,
                                                       stretch=self.stretchModel(
                                                           self, None, stampData[cutoutModel > 0]))

                if gridSpecs is None:
                    subplotAxes = mplplot.subplot(
                        2, 1, stampIndex + 1, projection=wcsObject)
                else:
                    subplotAxes = mplplot.subplot(
                        gridSpecs[stampIndex], projection=wcsObject)

                # Force stamps to plot within full wavelength range.
                subplotAxes.set_xlim(wcsObject.wcs_world2pix(StampPlotter.plottedWavelengthRange[0].value, 0, 1)[0],
                                     wcsObject.wcs_world2pix(StampPlotter.plottedWavelengthRange[1].value, 0, 1)[0])

                # Using aspect='auto' will force the subplots to stretch to fill the
                # available space.
                mplplot.imshow(stampData,
                               origin='lower',
                               interpolation='nearest',
                               norm=norm,
                               cmap=colourMap,
                               aspect='auto')

                mplplot.xlabel('Wavelength ({})'.format(wavelengthUnit))
                mplplot.ylabel('Cross-dispersion ({})'.format(xDispUnit))
                mplplot.title('{}Field {}, Object {}: \nDrizzled stamp for G{}{}'.format(titlePrefix, self.targetPar,
                                                                                         self.targetObject,
                                                                                         grism,
                                                                                         titleSuffix))
                mplplot.grid(color='white', ls='solid')

                if zerothOrderData is not None and zerothOrderData[grism] is not None:
                    # print ('Setting stamp path to {}'.format(self.stampPaths[grism]))
                    # zerothOrderData[grism].setDrizzledStampFilePath(self.stampPaths[grism])
                    # midpointWavelength = 0.5*(StampPlotter.grismRanges[grism][0] + StampPlotter.grismRanges[grism][1]).to(astrounits.angstrom).value
                    # zerothOrderData[grism].getWavelengthZeroOrderFlag(midpointWavelength)
                    self.plotZerothOrders(
                        zerothOrderData, subplotAxes, grism, wcsObject)

                # if modelRect is not None :
                #     mplplot.gca().add_patch(modelRect)

            mplplot.tight_layout(h_pad=5.0)
            if savePath is not None:
                mplplot.savefig(savePath, dpi=300, bbox_inches='tight')
                mplplot.close()
            else:
                return allSubPlotAxes
        else:
            print(
                'The loadDrizzledStamps(...) method must be called before drizzled stamps can be plotted.')

    def getDirectCutoutExtentsArcsec(self, grism):
        if grism in self.directCutoutExtentsArcsec and self.directCutoutExtentsArcsec[grism] is not None:
            return self.directCutoutExtentsArcsec[grism]
        else:
            print(('Direct cutout size in arc-seconds not available for G{} (F{}W) '
                   'Note that the direct cutout must have been plotted before this value '
                   'is computed.').format(grism, StampPlotter.grismToFilterMap[grism]))
            return None

    def plotDirectCutouts(self, savePath=None, colourMap='viridis', gridSpecs=None):
        if self.directCutouts is not None:
            allSubPlotAxes = []
            if gridSpecs is None:
                mplplot.figure(figsize=(10, 5))
            for stampIndex, (grism, cutoutData) in enumerate(self.directCutouts.items()):
                if gridSpecs is None:
                    subplotAxes = mplplot.subplot(2, 1, stampIndex + 1)
                else:
                    subplotAxes = mplplot.subplot(gridSpecs[stampIndex])

                allSubPlotAxes.append(subplotAxes)

                if cutoutData is None:
                    subplotAxes.text(0.5,
                                     0.5,
                                     'Field {}\nObject {}\nGrism G{}\nNO DATA AVAILABLE.'.format(self.targetPar,
                                                                                                 self.targetObject,
                                                                                                 grism),
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     fontsize='large',
                                     transform=subplotAxes.transAxes)
                    subplotAxes.set_axis_off()
                    continue
                if np.all(cutoutData < 0):
                    subplotAxes.text(0.5,
                                     0.5,
                                     'Field {}\nObject {}\nGrism G{}\nNO NONZERO DATA AVAILABLE.'.format(self.targetPar,
                                                                                                         self.targetObject,
                                                                                                         grism),
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     fontsize='large',
                                     transform=subplotAxes.transAxes)
                    subplotAxes.set_axis_off()
                    continue

                if self.useOptimalZScaleForDirectCutout:
                    (vMin, vMax), interval = self.computeDefaultZRange(
                        cutoutData, np.zeros_like(cutoutData))
                    norm = astromplnorm.ImageNormalize(cutoutData,
                                                       interval=interval,
                                                       stretch=self.stretchModel(
                                                           self, None, cutoutData))
                else:
                    norm = astromplnorm.ImageNormalize(cutoutData,
                                                       interval=astrovis.AsymmetricPercentileInterval(
                                                           0, 99.5),
                                                       stretch=astrovis.LinearStretch(),
                                                       clip=True)

                mplplot.imshow(cutoutData,
                               origin='lower',
                               interpolation='nearest',
                               cmap=colourMap,
                               norm=norm)

                mplplot.xlabel('X (pixels)')
                mplplot.ylabel('Y (pixels)')
                mplplot.title('Field {}, Object {}:\nDirect cutout for F{} (G{})'.format(self.targetPar,
                                                                                         self.targetObject,
                                                                                         self.getDirectFilterForGrism(
                                                                                             grism),
                                                                                         grism))
                arcsecYAxis = subplotAxes.twinx()
                self.directCutoutExtentsArcsec[grism] = (tuple((np.array(subplotAxes.get_xlim()) - 0.5 *
                                                                np.sum(subplotAxes.get_xlim())) * self.directHdus[grism][1]['IDCSCALE']),
                                                         tuple((np.array(subplotAxes.get_ylim()) - 0.5 *
                                                                np.sum(subplotAxes.get_ylim())) * self.directHdus[grism][1]['IDCSCALE']))
                if self.verboseOutput:
                    print(grism, self.getDirectCutoutExtentsArcsec(grism))
                    print(subplotAxes.get_ylim(),
                          np.array(subplotAxes.get_ylim()),
                          np.array(subplotAxes.get_ylim()) *
                          self.directHdus[grism][1]['IDCSCALE'],
                          np.array(subplotAxes.get_ylim()) * self.directHdus[grism][1]['IDCSCALE'])

                arcsecYAxis.set_ylim(
                    *self.getDirectCutoutExtentsArcsec(grism)[1])
                arcsecYAxis.set_ylabel('$\Delta Y$ (arcsec)')
                mplplot.grid(color='white', ls='solid')

            try:
                mplplot.tight_layout()
            except ValueError as e:
                print('Error attempting tight_layout for: Field {}, Object {} ({})'.format(self.targetPar,
                                                                                           self.targetObject,
                                                                                           e))
                return
            if savePath is not None:
                mplplot.savefig(savePath, dpi=300, bbox_inches='tight')
                mplplot.close()
            else:
                return allSubPlotAxes
        else:
            print(
                'The loadDirectCutouts(...) method must be called before direct cutouts can be plotted.')

    def plotExtraDirectCutout(self,
                              dataHdu,
                              position,
                              gridSpecs,
                              dataDescription=None,
                              sizeArcsec=None,
                              savePath=None,
                              colourMap='viridis',
                              matchExtentforGrism=141):
        if dataHdu is not None:
            try:
                cutoutSize = sizeArcsec if sizeArcsec is not None else tuple([(axisExtent[1] - axisExtent[0]) for axisExtent in self.getDirectCutoutExtentsArcsec(
                    matchExtentforGrism)])
                cutoutSize *= astrounits.arcsec
                wcs = astrowcs.WCS(header=dataHdu.header)
                cutout = astronddata.Cutout2D(
                    data=dataHdu.data, position=position, size=cutoutSize, wcs=wcs)

                cutoutData = cutout.data
            except astrondutils.NoOverlapError as e:
                print('Error generating cutout for Field {}, Object {}{}: {}'.format(self.targetPar,
                                                                                   self.targetObject,
                                                                                   ', {}'.format(dataDescription) if dataDescription is not None else '',
                                                                                   repr(e)))
                cutoutData = None
            except ValueError as e :
                print('Error generating cutout for Field {}, Object {}{}: {}'.format(self.targetPar,
                                                                                   self.targetObject,
                                                                                   ', {}'.format(dataDescription) if dataDescription is not None else '',
                                                                                   repr(e)))
                cutoutData = None

        subplotAxes = mplplot.subplot(gridSpecs)

        if dataHdu is None or cutoutData is None or cutoutData.size == 0:
            subplotAxes.text(0.5,
                             0.5,
                             'Field {}\nObject {}{}\nNO DATA AVAILABLE.'.format(self.targetPar,
                                                                                self.targetObject,
                                                                                '\n{}'.format(
                                                                                dataDescription) if dataDescription is not None else ''),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize='large',
                             transform=subplotAxes.transAxes)
            subplotAxes.set_axis_off()
            return subplotAxes
        if np.all(cutoutData < 0):
            subplotAxes.text(0.5,
                             0.5,
                             'Field {}\nObject {}{}\nNO NONZERO DATA AVAILABLE.'.format(self.targetPar,
                                                                                        self.targetObject,
                                                                                        '\n{}'.format(
                                                                                            dataDescription) if dataDescription is not None else '',
                                                                                        ),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize='large',
                             transform=subplotAxes.transAxes)
            subplotAxes.set_axis_off()
            return subplotAxes

        if self.useOptimalZScaleForDirectCutout:
            (vMin, vMax), interval = self.computeDefaultZRange(
                cutoutData, np.zeros_like(cutoutData))
            norm = astromplnorm.ImageNormalize(cutoutData,
                                               interval=interval,
                                               stretch=self.stretchModel(
                                                   self, None, cutoutData))
        else:
            norm = astromplnorm.ImageNormalize(cutoutData,
                                               interval=astrovis.AsymmetricPercentileInterval(
                                                   0, 99.5),
                                               stretch=astrovis.LinearStretch(),
                                               clip=True)

        mplplot.imshow(cutoutData,
                       origin='lower',
                       interpolation='nearest',
                       cmap=colourMap,
                       norm=norm)

        mplplot.xlabel('X (pixels)')
        mplplot.ylabel('Y (pixels)')
        mplplot.title('Field {}, Object {}:\nDirect cutout {}'.format(self.targetPar,
                                                                      self.targetObject,
                                                                      dataDescription if dataDescription is not None else ''))
        arcsecYAxis = subplotAxes.twinx()
        # print(cutoutSize.value, (cutoutSize.value - 0.5*np.sum(cutoutSize.value)))
        arcsecYAxis.set_ylim(-0.5 *
                             cutoutSize.value[1], 0.5 * cutoutSize.value[1])

        arcsecYAxis.set_ylabel('$\Delta Y$ (arcsec)')
        mplplot.grid(color='white', ls='solid')

        try:
            mplplot.tight_layout()
        except ValueError as e:
            print('Error attempting tight_layout for: Field {}, Object {} ({})'.format(self.targetPar,
                                                                                       self.targetObject,
                                                                                       e))
            return
        if savePath is not None:
            mplplot.savefig(savePath, dpi=300, bbox_inches='tight')
            mplplot.close()
        else:
            return subplotAxes

    def getDrizzledStampData(self, grism):
        if self.stampHdus is not None:
            return self.stampHdus['SCI'][grism][0] if self.stampHdus['SCI'][grism] is not None else None
        else:
            print(
                'The loadDrizzledStamps(...) method must be called before drizzled stamp data can be returned.')

    def getDrizzledStampHeader(self, grism):
        if self.stampHdus is not None:
            return self.stampHdus['SCI'][grism][1] if self.stampHdus['SCI'][grism] is not None else None
        else:
            print('The loadDrizzledStamps(...) method must be called before drizzled stamp header can be returned.')

    def getDirectCutout(self, grism):
        if self.directCutouts is not None:
            return self.directCutouts[grism] if self.directCutouts[grism] is not None else None
        else:
            print(
                'The loadDirectCutouts(...) method must be called before direct cutouts can be returned.')

    def getThresholdedDirectCutout(self, grism):
        skyKeyword = 'MDRIZSKY'
        if self.directHdus is not None:
            return self.directHdus[grism][1][skyKeyword]
        else:
            print('The loadDirectCutouts(...) method must be called before sky-subtracted direct cutouts can be returned.')

    def makeStandardGridSpecForDrizzledStamps(self):
        # generate a standard grid specification that includes positions for
        # the stamp imagegridSpec
        fullGridSpec = mplgs.GridSpec(
            3, 2, width_ratios=[4, 1], height_ratios=[4, 1, 1])
        # return the portion of the grid specification into which the stamps
        # should be plotted.
        return (fullGridSpec[1, 0], fullGridSpec[2, 0])

    def makeStandardGridSpecForDirectCutouts(self):
        # generate a standard grid specification that includes positions for
        # the stamp imagegridSpec
        fullGridSpec = mplgs.GridSpec(
            3, 2, width_ratios=[4, 1], height_ratios=[4, 1, 1])
        # return the portion of the grid specification into which the stamps
        # should be plotted.
        return (fullGridSpec[1, 1], fullGridSpec[2, 1])

    def computeDefaultZRange(self,
                             plottableData,
                             contaminationMap,
                             clipLevels=(5, 3)):
        contamStdDev = np.std(contaminationMap)
        contamThreshold = np.percentile(contaminationMap, 70)
        # This percentile can be zero for targets with effectively zero
        # contamination in such cases, fall back to the mean contamination
        # level
        contamThreshold = np.average(
            contaminationMap) if contamThreshold <= 0 else contamThreshold

        selectionMask = (contaminationMap < contamThreshold) if (
            contamStdDev > 0) else np.ones_like(plottableData, dtype=int)

        maskSum = np.sum(selectionMask)
        if maskSum <= 0:
            raise UserWarning('In computeOptimalZRange: Sum of selection mask is zero or less ({mask_sum}, {contam_std_dev}, {contam_threshold}, {mean_contam})!'.format(
                mask_sum=maskSum, contam_std_dev=contamStdDev,
                contam_threshold=contamThreshold,
                mean_contam=np.average(contaminationMap)))

        normalizationInterval = astrovis.ZScaleInterval()
        vmin, vmax = normalizationInterval.get_limits(
            plottableData[selectionMask])

        return (vmin, vmax), normalizationInterval
