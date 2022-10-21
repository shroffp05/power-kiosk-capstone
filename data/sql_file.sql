SELECT 
	cl.contractLocationID
	, vcl.plc
	, vcl.addDate
	, vcl.period
	, vcl.usage
	, c.status
	, cl.city
	, cl.stateID
	, ct.*
FROM 
(
	SELECT 
		SUM(usage) as totalYearlyUsage
		, contractLocationId
		, contractNum
	FROM (
		SELECT 
			contractNum,
			contractLocationId,
			usage,
			RANK() OVER (PARTITION BY contractLocationId ORDER BY [period] DESC) ranking
		FROM (
			SELECT
				c.contractNum,
				h.contractLocationId,
				h.usage,
				h.[period],
				RANK() OVER (PARTITION BY h.contractLocationId, [period] ORDER BY h.[addDate] desc) ranking
			FROM ViewContractLocationUsageHistories h
			INNER JOIN contractLocation cl on cl.contractLocationID = h.contractLocationID
			INNER JOIN contract c on c.contractId = cl.contractId
			WHERE c.serviceTypeID = '297ed5063d424e7b013d429edf0d0006'	
			AND c.status IN (2,6) 
		) s
		WHERE s.ranking = 1
	) r
	WHERE r.ranking <= 12
	GROUP BY contractNum, contractLocationId
	HAVING SUM(usage) > 200000
) AS t 
JOIN ViewContractLocationUsageHistories vcl 
	ON vcl.contractLocationID = t.contractLocationID
JOIN contractLocation cl 
	ON cl.contractLocationID = t.contractLocationID
JOIN contract c 
	ON c.contractID = cl.contractID
JOIN customer ct 
	ON ct.customerID = c.customerID
WHERE cl.contractLocationID IN [{}]