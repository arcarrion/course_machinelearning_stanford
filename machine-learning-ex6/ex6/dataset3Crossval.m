function [bestC, bestSigma, crossvalError] = dataset3Crossval(Xval, yval)

CSet = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaSet = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

nCvals = length(CSet);
nSigmaVals = length(sigmaSet);

crossvalError = zeros(nCvals*nSigmaVals,1);

for iC = 1:nCvals
    Ctest = CSet(iC);
    
    for iSigma = 1:nSigmaVals
        sigmaTest = sigmaSet(iSigma);
        
        model= svmTrain(Xval, yval, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
        predictions = svmPredict(model, Xval);
        crossvalError((iC-1)*nSigmaVals + iSigma) = mean(double(predictions ~= yval)); 
    end
end

[val,index] = min(crossvalError);

indexBestC = ceil(index/nSigmaVals);
bestC = CSet(indexBestC);
bestSigma = sigmaSet(index-((indexBestC-1)*nSigmaVals));

end
