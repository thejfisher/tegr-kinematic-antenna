function pdf = PDFB(x,cdf)

pdf = LOESS([x cdf],2,8,0.6);
pdf = pdf(:,2);

pdf = pdf/sum(pdf);

end