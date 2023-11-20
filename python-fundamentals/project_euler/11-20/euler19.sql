
-- Solved in T-SQL because why not?!
DECLARE @start DATE = '1901-01-01'
DECLARE @end DATE = '2000-12-31'
DECLARE @counter INT = 1
DECLARE @sunday_counter INT = 0

WHILE DATEADD( DAY, @counter, @start ) < @end
BEGIN
    DECLARE @curr_date DATE = DATEADD( DAY, @counter, @start )
    IF DATEPART( DAY, @curr_date ) = 1 AND DATEPART( WEEKDAY, @curr_date ) = 1
    BEGIN
	   SET @sunday_counter = @sunday_counter + 1
    END
    SET @counter = @counter + 1
END

SELECT @sunday_counter AS [Number of Sundays]