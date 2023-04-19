# Seasonal / time series analysis notes

want to:
- decompose velocity time series (some sort of area mean - whole glacier? subset?) into trend, seasonal, residual
- perform harmonic regression/fourier analysis on time series data

but need to do some more organizing before its ready for that
currently have:
- velocity observations from image pairs (range of image separation times)
- indexed by midpoint
- longer image pair times --> movement occurred over more days (usually slower obs)
- shorter image pair times --> movement occurred over fewer days (usually faster obs)
- how to weight these so that they are treated 'equally' in regression?
