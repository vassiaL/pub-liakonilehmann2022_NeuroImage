function actionId = getActionIdFromKeyId_2(keyId)
% maps the input-device specific keyId onto an actionId

[~, ~, actionNames] = getTechEnums;
%mapping for a logitech keyboard:
switch keyId
    case {KbName('RightArrow'), KbName('k'), KbName('1!'), KbName('6^')} %right arrow or letter K
        actionId = 1;
    case {KbName('LeftArrow'), KbName('d'), KbName('4$'), KbName('9(')} %left arrow or letter D
        actionId = 2;
    otherwise
        actionId = actionNames.undefined;
end

end

